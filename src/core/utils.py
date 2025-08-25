"""-------- LLM Loader Utils --------"""

from typing import Dict, Any
import torch


def remove_model_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Removes the "model." prefix from the keys of a Pytorch state dict"""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.removeprefix("model.")
        new_state_dict[new_key] = value
    return new_state_dict
