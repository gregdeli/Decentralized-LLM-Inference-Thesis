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
    print(f"Successfully discovered bootstrap peer: {initial_peers[0]}")

    client = Client(
        model_path=model_path,
        host_maddrs=host_maddrs,
        initial_peers=initial_peers,
    )

    # prompt = "The name of the capital of France is"

    try:
        while True:
            prompt = input("\nEnter prompt: ")

            logger.info(f"Initiating text generation...")
            # logger.info(f"\n--------Prompt--------\n{prompt}")
            text = client.generate(prompt, max_new_tokens=10)

            if not text:
                input()
                continue

            print(f"--------Response--------\n{prompt + text}")

    except KeyboardInterrupt:
        print("\nExiting...")  # Ctrl+C


if __name__ == "__main__":
    main()
