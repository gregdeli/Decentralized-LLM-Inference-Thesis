from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Llama3(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
        is_client: bool,
        layers_to_load: Tuple[int, int]
    ) -> None:
        """
        Args:
            is_client: If it is True, the model is being loaded on a client node, meaning that the
                       embedding layer, some optinal transformer layers and the final norm and lm_head layers will be loaded.
                       If it is False, the model is being loaded on a server node, meaning that only num_layers transformer layer
                       will be loaded.
            num_layers: Defines the number of transformer to be loading in this node.
        """
        super().__init__()
        self.config = config
        self.is_client = is_client

        if layers_to_load is None:
            self.num_layers = 0
        else:
            self.num_layers = layers_to_load[1] - layers_to_load[0] + 1

        if is_client:
            self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        if self.num_layers > 0:
            self.layers = nn.ModuleDict(
                {str(block_idx): TransformerBlock(config, block_idx) for block_idx in range(layers_to_load[0], layers_to_load[1] + 1)}
            )
        if is_client:
            self.norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
            self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

        self.mask_cache: Optional[torch.Tensor] = None
        self.rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        self.max_seq_length = self.config["max_position_embeddings"]

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        # input_pos_maxp1: Optional[int] = None,  # Xrisimpopoieitai otan kanoume compile
    ) -> torch.Tensor:
        T = input_ids.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(device=input_ids.device)

        # Get the RoPE embeddings for the current sequence
        cos, sin = self.rope_cache
        if input_pos is None:  # prefill
            cos = cos[:T]
            sin = sin[:T]
        else:  # generation
            cos = cos[input_pos]
            sin = sin[input_pos]

        # Get the attention mask
        mask = self.mask_cache
        if mask is not None and T > 1:  # prefill
            mask = mask[:, :, :T, :T]
        else:
            mask = None

        # Forward pass
        x = self.embed_tokens(input_ids)

        if self.num_layers > 0:
            for block in self.layers.values():
                x = block(x, cos, sin, mask, input_pos)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def forward_client_initial(
        self,
        input_ids: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T = input_ids.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(device=input_ids.device)

        # Get the RoPE embeddings for the current sequence
        cos, sin = self.rope_cache
        if input_pos is None:  # prefill
            cos = cos[:T]
            sin = sin[:T]
        else:  # generation
            cos = cos[input_pos]
            sin = sin[input_pos]

        # Get the attention mask
        mask = self.mask_cache
        if mask is not None and T > 1:  # prefill
            mask = mask[:, :, :T, :T]
        else:
            mask = None

        # Forward pass
        x = self.embed_tokens(input_ids)

        # if self.num_layers > 0:
        #     for block in self.layers.values():
        #         x = block(x, cos, sin, mask, input_pos)
        return x

    def forward_client_final(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm(input)
        logits = self.lm_head(x)
        return logits

    def forward_server(
        self,
        input: torch.Tensor,
        seq_length: Optional[int],
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prefill:
            The seq_length must be sent to the server in order to get the correct range of positional embeddings
            in the prefill phase. The input_pos arguement remains None at this stage.
        Generation:
            In the generation phase the input_pos of the current token that is being generated must be sent
            to the server. The seq_length arguement remains None at this stage.
        """

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(device=input.device)

        T = seq_length
        # Get the RoPE embeddings for the current sequence
        cos, sin = self.rope_cache
        if input_pos is None:  # prefill
            cos = cos[:T]
            sin = sin[:T]
        else:  # generation
            cos = cos[input_pos]
            sin = sin[input_pos]

        # Get the attention mask
        mask = self.mask_cache
        if mask is not None and T > 1:  # prefill
            mask = mask[:, :, :T, :T]
        else:
            mask = None

        h = input
        if self.num_layers > 0:
            for block in self.layers.values():
                h = block(h, cos, sin, mask, input_pos)

        return h

    def build_rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config["head_dim"],
            device=device,
            base=self.config["rope_theta"],
        )

    def set_kv_cache(
        self,
        batch_size: int,
        max_seq_length: Optional[int] = None,
        device: Optional[torch.device] = "cpu",
        dtype: Optional[torch.dtype] = torch.get_default_dtype(),
    ) -> None:
        """
        Pre-allocates the K-V cache for each transformer block.
        """
        if max_seq_length is None:
            max_seq_length = self.max_seq_length

        # Initialize kv cache for all blocks
        if self.num_layers > 0:
            for block in self.layers.values():
                block.self_attn.kv_cache = block.self_attn.build_kv_cache(batch_size, max_seq_length, device, dtype)

        # Create the causal attention mask and cache it
        # Pairnei ligh wra auto
        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        if self.num_layers > 0:
            for block in self.layers.values():
                block.self_attn.kv_cache = None

    def post_init(self):
        # Tie the weights between the input embeddings and the ouput embeddings.
        if self.config.get("tie_word_embeddings", True) and self.is_client:
            self.lm_head.weight = self.embed_tokens.weight


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
        block_idx: int,
    ) -> None:
        super().__init__()

        self.input_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.self_attn = CausalSelfAttention(config, block_idx)
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.mlp = MLP(config)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x

        # Input LayerNorm + Self-Attention
        h = self.input_layernorm(x)
        h = self.self_attn(h, cos, sin, mask, input_pos)
        h = residual + h  # First residual connection

        # Post-Attention LayerNorm + MLP
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        h = residual + h  # Second residual connection

        return h


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Dict[str, Any], block_idx: int) -> None:
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.n_head = config["num_attention_heads"]
        self.n_kv_head = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.n_rep = self.n_head // self.n_kv_head
        bias = config.get("attention_bias", False)

        # q, k, v projections
        self.q_proj = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_head * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_head * self.head_dim, bias=bias)

        # Output projection
        self.o_proj = nn.Linear(self.n_head * self.head_dim, self.hidden_size, bias=bias)

        self.kv_cache: Optional[KVCache] = None  # Placeholder for the KV cache
        self.config = config
        self.block_idx = block_idx

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape  # Batch size, sequence length, embedding dimensionality

        # Calculate Q, K, V projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # KV Caching
        if self.kv_cache is not None:
            k, v = self.kv_cache(input_pos, k, v)

        # GQA: repeat K and V heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)  # Repeat elements of the tensor along the head dim
            v = v.repeat_interleave(self.n_rep, dim=1)

        # In the prefill phase attend only to the tokens in the prompt
        if input_pos is None:
            k = k[:, :, :T, :]
            v = v[:, :, :T, :]
        else:
            current_seq_len = input_pos.max() + 1
            k = k[:, :, :current_seq_len, :]
            v = v[:, :, :current_seq_len, :]

        # Scaled Dot-Product Attention
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)

        # Reshape and final projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)

        return y

    def build_kv_cache(
        self, batch_size: int, max_seq_length: int, device: Optional[torch.device] = None, dtype: Optional[torch.device] = None
    ) -> "KVCache":
        """
        Builds the K-V cache for this attention layer
        """
        k_shape = (batch_size, self.config["num_key_value_heads"], max_seq_length, self.config["head_dim"])
        v_shape = (batch_size, self.config["num_key_value_heads"], max_seq_length, self.config["head_dim"])
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)


class MLP(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]
        bias = config.get("mlp_bias", False)

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = F.silu(gate) * up
        return self.down_proj(x)


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization.
    """

    def __init__(self, size: int, eps: float = 1e-05) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        varience = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(varience + self.eps)
        return (self.weight * x).to(input_dtype)


class KVCache(nn.Module):
    """
    Buffers `k`, `v` have shape (batch_size, n_query_groups, max_seq_length, head_size)`.
    """

    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)

    def forward(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the K-V cache and returns the full cached tensors.
        """
        if input_pos is None:  # prefill phase
            seq_len = k.size(2)
            input_pos = torch.arange(0, seq_len, device=k.device)

        # Ensure correct dtype
        # Error: self.k = ... destroys pytorch buffer registration
        # if self.k.dtype != k.dtype:
        #     self.k = self.k.to(k.dtype)
        # if self.v.dtype != v.dtype:
        #     self.v = self.v.to(v.dtype)

        # Update cache
        self.k.index_copy_(2, input_pos, k)
        self.v.index_copy_(2, input_pos, v)

        return self.k, self.v


def build_mask_cache(max_seq_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Builds a causal attention mask for a given sequence length.
    """
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)


def build_rope_cache(seq_len: int, n_elem: int, device: Optional[torch.device] = None, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Builds the RoPE cache."""
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    seq_idx = torch.arange(seq_len, device=device, dtype=theta.dtype)
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)
    return torch.cos(idx_theta), torch.sin(idx_theta)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]
    x2 = x[..., head_size // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)
