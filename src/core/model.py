from typing import Dict, Any, Optional, Tuple
import time

from core.constants import GLOBAL_MAX_SEQ_LEN

import torch
import torch.nn as nn
import torch.nn.functional as F


class Llama3(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
        load_initial_layer: bool,
        layers_to_load: Tuple[int, int],
        load_output_layer: bool,
    ) -> None:
        """
        Args:
            is_client: If it is True, the model is being loaded on a client node, meaning that the
                       embedding layer, some optinal transformer layers and the final norm and lm_head layers will be loaded.
                       If it is False, the model is being loaded on a server node, meaning that only num_layers transformer layer
                       will be loaded.
            num_layers: Defines the number of transformer to be loading in this node.
        """
        # Dynamically calculate and set head_dim if its missing
        if "head_dim" not in config:
            config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]

        super().__init__()
        self.config = config
        self.output_layer_loaded = load_output_layer

        if layers_to_load is None:
            self.num_layers = 0
        else:
            self.num_layers = layers_to_load[1] - layers_to_load[0] + 1

        if load_initial_layer:
            self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        if self.num_layers > 0:
            self.layers = nn.ModuleDict(
                {
                    str(block_idx): TransformerBlock(config, block_idx)
                    for block_idx in range(layers_to_load[0], layers_to_load[1] + 1)
                }
            )
        if load_output_layer:
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
            raise ValueError(
                f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}."
            )

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
        # input_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        T = input_ids.size(1)
        if self.max_seq_length < T:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}."
            )

        is_cuda = input_ids.device.type == "cuda"

        if is_cuda:
            torch.cuda.synchronize(input_ids.device)

        embed_start_time = time.perf_counter()

        # Forward pass
        x = self.embed_tokens(input_ids)

        if is_cuda:
            torch.cuda.synchronize(input_ids.device)

        embed_delay = time.perf_counter() - embed_start_time

        return x, embed_delay

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
        seq_length: Optional[int] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Executes the forward pass through the server's assigned transformer layers.

        Args:
            input (torch.Tensor): The input hidden states tensor received from the predecessor node.
            seq_length (Optional[int]): The number of tokens in the current payload. Used to
                                        distinguish between the prefill phase (seq_length > 1)
                                        and the generation phase (seq_length == 1).
            input_pos (Optional[torch.Tensor]): A 1-D tensor containing the absolute sequence
                                                positions for the tokens in the current payload.
                                                - Prefill: Contains a sequence of positional indices mapping the new prompt.
                                                - Generation: Contains a single positional index for the newly generated token.

        Returns:
            torch.Tensor: The computed hidden states to be forwarded to the successor node,
                          or the final logits if this node holds the output layer.
        """

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(device=input.device)

        h = input

        is_cuda = input.device.type == "cuda"

        # Start timing the Transformer Layer

        if is_cuda:
            torch.cuda.synchronize(input.device)

        trans_start_time = time.perf_counter()

        if self.num_layers > 0:
            T = seq_length
            # Get the RoPE embeddings for the current sequence
            cos, sin = self.rope_cache
            cos = cos[input_pos]
            sin = sin[input_pos]

            # Get the attention mask
            mask = self.mask_cache
            if mask is not None and T > 1:  # prefill
                entire_seq_len = input_pos.max() + 1
                mask = mask[:, :, input_pos, :entire_seq_len]
            else:
                mask = None

            for block in self.layers.values():
                h = block(h, cos, sin, mask, input_pos)

        if is_cuda:
            torch.cuda.synchronize(input.device)
        trans_end_time = time.perf_counter()

        if self.output_layer_loaded:
            x = self.norm(h)
            h = self.lm_head(x)

        if is_cuda:
            torch.cuda.synchronize(input.device)

        end_time = time.perf_counter()

        transformer_layer_delay = trans_end_time - trans_start_time if self.num_layers > 0 else 0.0
        output_layer_delay = end_time - trans_end_time if self.output_layer_loaded else 0.0

        return h, transformer_layer_delay, output_layer_delay

    def build_rope_cache(
        self, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cache(
            seq_len=GLOBAL_MAX_SEQ_LEN,
            n_elem=self.config["head_dim"],
            device=device,
            base=self.config["rope_theta"],
        )

    def client_has_cache(self, client_id: str) -> bool:
        """Check if the given client has a kv cache allocated"""
        if self.num_layers > 0:
            for block in self.layers.values():
                if client_id not in block.self_attn.kv_caches:
                    return False
            return True

        return False

    def add_client_cache(
        self,
        client_id: str,
        batch_size: int = 1,
        max_seq_length: Optional[int] = GLOBAL_MAX_SEQ_LEN,
        device: Optional[torch.device] = "cpu",
        dtype: Optional[torch.dtype] = torch.get_default_dtype(),
    ) -> None:
        """Allocate a KV Cache for the new client"""
        # First reallocate the caches of the previous clients so that their local max_seq_lenths add up to GLOBAL_MAX_SEQ_LEN
        if self.num_layers > 0:
            for block in self.layers.values():
                for client in block.self_attn.kv_caches.keys():
                    if max_seq_length != block.self_attn.kv_caches[client].max_seq_length:
                        prev_max_seq_length = block.self_attn.kv_caches[client].max_seq_length

                        block.self_attn.kv_caches[client] = block.self_attn.rebuild_kv_cache(
                            batch_size, max_seq_length, prev_max_seq_length, device, dtype
                        )

        # Allocate the new client's kv cache
        if self.num_layers > 0:
            for block in self.layers.values():
                block.self_attn.kv_caches[client_id] = block.self_attn.build_kv_cache(
                    batch_size, max_seq_length, device, dtype
                )

        # Ensure mask cache exists
        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def remove_client_cache(self, client_id: str) -> None:
        """Frees the KV cache memory of a certain client_id"""
        if self.num_layers > 0:
            for block in self.layers.values():
                if client_id in block.self_attn.kv_caches:
                    del block.self_attn.kv_caches[client_id]

    def reallocate_caches(
        self,
        batch_size: int = 1,
        max_seq_length: Optional[int] = GLOBAL_MAX_SEQ_LEN,
        device: Optional[torch.device] = "cpu",
        dtype: Optional[torch.dtype] = torch.get_default_dtype(),
    ) -> None:
        """
        Realocate the caches of the remaining clients so that each of their caches
        has a max_seq_length of GLOBAL_MAX_SEQ_LEN // num_clients
        """
        if self.num_layers > 0:
            for block in self.layers.values():
                for client_id in block.self_attn.kv_caches.keys():
                    prev_max_seq_length = block.self_attn.kv_caches[client_id].max_seq_length
                    block.self_attn.kv_caches[client_id] = block.self_attn.rebuild_kv_cache(
                        batch_size, max_seq_length, prev_max_seq_length, device, dtype
                    )

    def client_cache_reallocated(
        self,
        client_id: str,
    ) -> bool:
        cache_reallocated = False

        if self.num_layers > 0:
            for block in self.layers.values():
                client_cache = block.self_attn.kv_caches.get(client_id)
                if (
                    client_cache
                    and client_cache.prev_max_seq_length
                    and client_cache.max_seq_length != client_cache.prev_max_seq_length
                ):
                    # update the prev_max_seq_length
                    client_cache.prev_max_seq_length = client_cache.max_seq_length
                    cache_reallocated = True
                else:
                    return False

        return cache_reallocated

    def is_client_cache_fresh(self, client_id: str) -> bool:
        """Checks if the KV caches for a given client_id have not yet been written to"""
        if self.num_layers <= 0:
            return False

        for block in self.layers.values():
            client_cache = block.self_attn.kv_caches.get(client_id)
            
            # If the cache is missing from any loaded layer, it is not properly initialized.
            if client_cache is None:
                return False
            
            # KVCache initializes 'k' and 'v' buffers with torch.zeros. 
            if torch.any(client_cache.k) or torch.any(client_cache.v):
                return False

        #It is fresh.
        return True

    def set_active_client(self, client_id: str) -> None:
        """Rotates the active KV cache for the upcoming forward pass."""
        if self.num_layers > 0:
            for block in self.layers.values():
                block.self_attn.active_client_id = client_id

    def clear_all_kv_caches(self) -> None:
        """Clears all caches for all clients."""
        self.mask_cache = None
        if self.num_layers > 0:
            for block in self.layers.values():
                block.self_attn.kv_caches.clear()
                block.self_attn.active_client_id = None

    def get_kv_cache_memory_sizes(self) -> Dict[str, float]:
        """Returns the size of each client's KV Cache in MB"""
        kv_cache_memories = {}

        if self.num_layers > 0:
            for block in self.layers.values():
                for client_id in block.self_attn.kv_caches.keys():
                    kv_cache = block.self_attn.kv_caches[client_id]
                    if kv_cache is not None:
                        if client_id not in kv_cache_memories:
                            kv_cache_memories[client_id] = 0.0

                        client_cache_bytes = kv_cache.k.nelement() * kv_cache.k.element_size()
                        client_cache_bytes += kv_cache.v.nelement() * kv_cache.v.element_size()
                        kv_cache_memories[client_id] += client_cache_bytes / (1024 * 1024)

        return kv_cache_memories

    def get_client_kv_cache_memory_size(self, client_id: int) -> float:
        """Returns the size of a client's KV Cache in MB"""
        total_bytes = 0

        if self.num_layers > 0:
            for block in self.layers.values():
                kv_cache = block.self_attn.kv_caches[client_id]
                if kv_cache is not None:
                    total_bytes += kv_cache.k.nelement() * kv_cache.k.element_size()
                    total_bytes += kv_cache.v.nelement() * kv_cache.v.element_size()

        total_mb = total_bytes / (1024 * 1024)
        return total_mb

    def post_init(self):
        # Tie the weights between the input embeddings and the output embeddings
        # only if both layers are loaded on this specific node.
        if self.config.get("tie_word_embeddings", True):
            if hasattr(self, "embed_tokens") and hasattr(self, "lm_head"):
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
    def __init__(self, config: Dict[str, Any], block_idx: int):
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

        self.kv_caches: Dict[str, KVCache] = {}  # Dict of KV Caches, key is the client address
        self.active_client_id: Optional[str] = None

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
        if self.active_client_id is not None and self.active_client_id in self.kv_caches:
            active_cache = self.kv_caches[self.active_client_id]
            k, v = active_cache(input_pos, k, v)

        current_seq_len = input_pos.max() + 1
        k = k[:, :, :current_seq_len, :]
        v = v[:, :, :current_seq_len, :]

        # GQA: repeat K and V heads
        if self.n_rep > 1:
            k = k.repeat_interleave(
                self.n_rep, dim=1
            )  # Repeat elements of the tensor along the head dim
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled Dot-Product Attention
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)

        # Reshape and final projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)

        return y

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.device] = None,
    ) -> "KVCache":
        """
        Builds the K-V cache for this attention layer
        """
        k_shape = (
            batch_size,
            self.config["num_key_value_heads"],
            max_seq_length,
            self.config["head_dim"],
        )
        v_shape = (
            batch_size,
            self.config["num_key_value_heads"],
            max_seq_length,
            self.config["head_dim"],
        )
        return KVCache(k_shape, v_shape, max_seq_length=max_seq_length, device=device, dtype=dtype)

    def rebuild_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        prev_max_seq_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.device] = None,
    ) -> "KVCache":
        """
        Rebuilds the K-V cache for this attention layer
        """
        k_shape = (
            batch_size,
            self.config["num_key_value_heads"],
            max_seq_length,
            self.config["head_dim"],
        )
        v_shape = (
            batch_size,
            self.config["num_key_value_heads"],
            max_seq_length,
            self.config["head_dim"],
        )
        return KVCache(
            k_shape,
            v_shape,
            max_seq_length=max_seq_length,
            prev_max_seq_length=prev_max_seq_length,
            device=device,
            dtype=dtype,
        )


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
        max_seq_length: int,
        prev_max_seq_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.prev_max_seq_length = prev_max_seq_length
        self.max_seq_length = max_seq_length
        self.register_buffer(
            "k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False
        )

    def forward(
        self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the K-V cache and returns the full cached tensors.
        """

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


def build_rope_cache(
    seq_len: int, n_elem: int, device: Optional[torch.device] = None, base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
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
