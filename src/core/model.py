from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Llama3(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(TransformerBlock(config, block_idx) for block_idx in range(config["num_hidden_layers"]))
        self.norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

        self.mask_cache: Optional[torch.Tensor] = None
        self.max_seq_length = self.config["max_position_embeddings"]

        self.post_init()

    def set_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Pre-allocates the K-V cache for each transformer block.
        """
        # Initialize kv cache for all blocks
        for block in self.layers:
            block.self_attn.kv_cache = block.self_attn.build_kv_cache(batch_size, max_seq_length, device, dtype)

        # Create the causal attention mask and cache it
        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def post_init(self):
        # Tie the weights between the input embeddings and the ouput embeddings.
        if self.config.get("tie_word_embeddings", True):
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


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Dict[str, Any], block_idx: int) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        n_head = config["num_attention_heads"]
        n_kv_head = config["num_key_value_heads"]
        head_dim = config["head_dim"]
        bias = config.get("attention_bias", False)

        # q, k, v projections
        self.q_proj = nn.Linear(hidden_size, n_head * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, n_kv_head * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, n_kv_head * head_dim, bias=bias)

        # Output projection
        self.o_proj = nn.Linear(n_head * head_dim, hidden_size, bias=bias)

        self.kv_cache: Optional[KVCache] = None  # Placeholder for the KV cache
        self.config = config
        self.block_idx = block_idx

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

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-05, add_unit_offset: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset


class KVCache(nn.Module):
    """
    Buffers `k`, `v` have shape
    `(batch_size, n_query_groups, max_seq_length, head_size)`.
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


def build_mask_cache(max_seq_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Builds a causal attention mask for a given sequence length.
    """
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)
