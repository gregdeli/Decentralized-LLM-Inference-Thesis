from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import json
import torch

from core.constants import *


logger = logging.getLogger(__name__)

"""-------------- LLM Loader Utils --------------"""

def get_dtype_from_config(config: Dict[str, Any]) -> torch.dtype:
    dtype_str = config.get("torch_dtype")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32
    }
    return dtype_map.get(dtype_str, torch.float32)


def get_relevant_safetensor_files(weight_map: Dict[str, str]):
    return list(set(weight_map.values()))


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


# Llama 3.2 1B -> transformer_layer_params = 60821504
# Llama 3.2 3B -> transformer_layer_params = 100669440
# Llama 3.1 8B -> transformer_layer_params = 218112000
def calculate_transformer_params(model_path: Path) -> int:
    """Calculates the total number of parameters for a single Transformer Layer for Llama 3"""
    config_path = model_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # Dynamically calculate and set head_dim if its missing
    if "head_dim" not in config:
            config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]

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

def calculate_final_output_params(model_path: Path) -> int:
    """Calculates the total number of parameters for the final RMS Norm and Linear Layer"""
    config_path = model_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # Dynamically calculate and set head_dim if its missing
    if "head_dim" not in config:
            config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]

    hidden_size = config["hidden_size"]
    vocab_size = config["vocab_size"]

    # Final RMS Norm Parameters
    final_norm_params = hidden_size

    # Linear Output Layer (lm head) Parameters
    lm_head_params = hidden_size * vocab_size

    return final_norm_params + lm_head_params

def update_config_layer_param_count(model_path: Path, param_count: int) -> None:
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)

    config["transformer_layer_params"] = param_count


    with open(f"{model_path}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Updated {model_path}/config.json with transformer_layer_params: {param_count}")


def update_config_final_output_param_count(model_path: Path, param_count: int) -> None:
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)

    config["final_output_params"] = param_count


    with open(f"{model_path}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Updated {model_path}/config.json with final_output_params: {param_count}")

def update_config_total_param_count(model_path: Path, param_count: int) -> None:
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)

    config["num_total_params"] = param_count


    with open(f"{model_path}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Updated {model_path}/config.json with num_total_params: {param_count}")


def calculate_model_size_mb(config: Dict[str, Any]) -> Tuple[float, float, float]:
    """Calculates the total size of the model when loaded in MB"""
    if "head_dim" not in config:
        config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]

    layer_params = config.get("transformer_layer_params")
    final_output_params = config.get("final_output_params")

    param_dtype = get_dtype_from_config(config)
    bytes_per_param = param_dtype.itemsize
    
    # Transformer Layer size
    layer_param_mem_size_mb = (layer_params * bytes_per_param) / (1024 * 1024)
    layer_kv_cache_mem_size_mb = (2 * config.get("num_key_value_heads") * GLOBAL_MAX_SEQ_LEN * config.get("head_dim") * bytes_per_param) / (1024 * 1024)
    layer_mem_size_mb = layer_param_mem_size_mb + layer_kv_cache_mem_size_mb

    # Final Output Layer size
    output_mem_size_mb = final_output_params * bytes_per_param / (1024 * 1024)

    num_total_layers = config.get("num_hidden_layers")
    model_size = num_total_layers * layer_mem_size_mb + output_mem_size_mb
    
    return (model_size, layer_mem_size_mb, output_mem_size_mb)


def can_load(
    config: Dict[str, Any],
    avail_mem: float,
    avail_vram: float,
    output_layer_memory_tle: float,
    num_layers: Optional[int] = None,
    layers: Optional[Tuple[int, int]] = None,
    output_layer: Optional[bool] = None,
) -> bool:
        """Checks if a node can load a certain number of transformer layers based on avail_mem or avail_vram"""
        num_layers = num_layers if num_layers else (layers[1] - layers[0] + 1) if layers else 0
        max_num_layers = mem_to_num_layers(config, avail_mem, avail_vram)

        if num_layers > 0:
            if max_num_layers < num_layers:
                return False
            
            if output_layer:
                max_num_layers -= num_layers
                return max_num_layers >= output_layer_memory_tle
            
            return True

        return max_num_layers >= output_layer_memory_tle

def mem_to_num_layers(
    config: Dict[str, Any],
    avail_mem: Optional[float],
    avail_vram: Optional[float],
) -> int:
    """Calculates how many transformer layers fit in the given available Memory/VRAM"""
    
    if "head_dim" not in config:
        config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]

    layer_params = config.get("transformer_layer_params")

    param_dtype = get_dtype_from_config(config)
    bytes_per_param = param_dtype.itemsize
    
    layer_param_mem_size_mb = (layer_params * bytes_per_param) / (1024 * 1024)
    layer_kv_cache_mem_size_mb = (2 * config.get("num_key_value_heads") * GLOBAL_MAX_SEQ_LEN * config.get("head_dim") * bytes_per_param) / (1024 * 1024)
    layer_mem_size_mb = layer_param_mem_size_mb + layer_kv_cache_mem_size_mb

    if avail_vram:
        available = avail_vram - RESERVED_MEM_MB
    else:
        available = avail_mem - RESERVED_MEM_MB

    if available <= 0:
        return 0

    max_num_layers = int(round(available / layer_mem_size_mb))
    return min(max_num_layers, config.get("num_hidden_layers"))

