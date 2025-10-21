from typing import Dict, Any
import logging
from pathlib import Path
import json


logger = logging.getLogger(__name__)

"""-------------- LLM Loader Utils --------------"""


def remove_model_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Removes the "model." prefix from the keys of a Pytorch state dict"""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.removeprefix("model.")
        new_state_dict[new_key] = value
    return new_state_dict


# Llama 3.2 -> total_transformer_layer_params = 60821504
def calculate_transformer_params(model_path_str: str) -> int:
    """Calculates the total number of parameters for a single Transformer Layer for Llama 3.2"""
    model_path = Path(model_path_str)
    config_path = model_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    num_heads = config["num_attention_heads"]
    num_k_v_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]

    # Total Attention Parameters
    W_q_params = hidden_size * (num_heads * head_dim)
    W_k_v_params = 2 * (hidden_size * (num_k_v_heads * head_dim))  # k_proj and v_proj
    W_o_params = (num_heads * head_dim) * hidden_size

    total_attention_params = W_q_params + W_k_v_params + W_o_params

    # Total MLP Parameters
    W_gate_up_down_params = hidden_size * intermediate_size
    total_mlp_params = 3 * W_gate_up_down_params

    # Total RMSNorm Params
    total_norm_params = 2 * hidden_size  # input_layer_norm and post_attention_layernorm

    return total_attention_params + total_mlp_params + total_norm_params
