from typing import Dict, Any
import torch
import grpc
import time
import logging
from pathlib import Path
import json

from core.remote import nodeservice_pb2_grpc, nodeservice_pb2

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
    W_k_v_params = 2* (hidden_size * (num_k_v_heads * head_dim)) # k_proj and v_proj
    W_o_params = (num_heads * head_dim) * hidden_size

    total_attention_params = W_q_params + W_k_v_params + W_o_params

    # Total MLP Parameters
    W_gate_up_down_params = hidden_size * intermediate_size
    total_mlp_params = 3 * W_gate_up_down_params

    # Total RMSNorm Params
    total_norm_params = 2 * hidden_size # input_layer_norm and post_attention_layernorm

    return total_attention_params + total_mlp_params + total_norm_params



"""-------------- GRPC Connection Utils --------------"""

def get_bootstrap_peer_address(address: str, attempts: int = 5) -> str | None:
    """
    Connects to a bootstrap gRPC server, retrieves its P2P multiaddress, and closes the connection.
    """
    for attempt in range(attempts):
        try:
            logger.info(f"Attempting to discover bootstrap peer at {address} (Attempt {attempt + 1})...")
            with grpc.insecure_channel(address) as channel:
                stub = nodeservice_pb2_grpc.NodeServiceStub(channel)
                response = stub.GetPeerMultiaddr(nodeservice_pb2.Empty(), timeout=5)
                
                if response and response.multiaddr:
                    return response.multiaddr
                
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                logger.info("Bootstrap node not ready yet, retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"An unexpected gRPC error occurred while contacting bootstrap node: {e}")
                return None
    
    print(f"FATAL: Could not connect to bootstrap node at {address} after {attempts} attempts.")
    return None

