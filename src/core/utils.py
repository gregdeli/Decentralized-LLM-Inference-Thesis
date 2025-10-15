from typing import Dict, Any
import torch
import grpc
import time
import logging

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

