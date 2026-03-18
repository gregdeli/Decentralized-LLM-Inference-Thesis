"""
A minimal CLI test script that discovers a bootstrap peer, creates a Client, 
and generates text from a hard-coded prompt.
"""

import os
from pathlib import Path
import time
import grpc
import logging

from client.client import Client
from core.remote import nodeservice_pb2, nodeservice_pb2_grpc
from core.remote.utils import get_bootstrap_peer_address

logger = logging.getLogger(__name__)


def main():
    model_path_str = os.getenv("MODEL_PATH")
    model_path = Path(model_path_str)

    host_maddrs = os.getenv("HOST_MADDRS")
    bootstrap_node_addr_str = os.getenv("BOOTSTRAP_NODE_ADDR")

    # Connect to the bootstrap node to get its p2p Multiaddress
    bootstrap_peer_addr = get_bootstrap_peer_address(bootstrap_node_addr_str, attempts=10)

    if not bootstrap_peer_addr:
        return

    initial_peers = [bootstrap_peer_addr]
    logger.info(f"Successfully discovered bootstrap peer: {initial_peers[0]}")

    client = Client(
        model_path=model_path,
        host_maddrs=host_maddrs,
        initial_peers=initial_peers,
    )

    prompt = "The capital of France is"

    # prompt = "Paris is the capital of"

    # prompt = "The capital of Greece is"

    # prompt = "A professional email from an employee to their boss about being sick:\n\nSubject: Out of Office Today - Unwell\n\nHi [Boss's Name],\n\nI am writing to inform you that"

    logger.info(f"Initiating text generation...")
    logger.info(f"\n--------Prompt--------\n{prompt}")

    for _ in range(1):
        text = client.generate(prompt, max_new_tokens=2)

        if text is not None:
            print(f"--------Response--------\n{prompt + text}")


if __name__ == "__main__":
    main()
