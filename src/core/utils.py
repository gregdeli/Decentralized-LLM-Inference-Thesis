from typing import Dict, Any
import logging
from pathlib import Path
import json


logger = logging.getLogger(__name__)

"""-------------- LLM Loader Utils --------------"""


def get_relevant_safetensor_files(weight_map: Dict[str, str]):
    return list(set(weight_map.values()))
    # relevant_files = set()
    # for filename in weight_map.values():
    #     relevant_files.add(filename)

    # return list(relevant_files)


def remove_model_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Removes the "model." prefix from the keys of a Pytorch state dict"""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.removeprefix("model.")
        new_state_dict[new_key] = value
    return new_state_dict


def is_instruct_model(model_path: Path) -> bool:
    """
    Checks if a model is instruction-tuned by looking for a
    chat template in its tokenizer_config.json.
    """
    tokenizer_config_path = model_path / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        False

    with open(tokenizer_config_path, "r") as f:
        config = json.load(f)

    if config.get("chat_template"):
        return True
    else:
        return False


# Llama 3.2 1B -> total_transformer_layer_params = 60821504
# Llama 3.2 3B -> total_transformer_layer_params = 100669440
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
