import os
from dotenv import load_dotenv
from pathlib import Path
import logging

from client.client import Client
from core.remote.utils import get_bootstrap_peer_address

logger = logging.getLogger(__name__)


def main():
    load_dotenv()
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
        host_maddrs=[host_maddrs],
        initial_peers=initial_peers,
    )

    try:
        conversation = ""  # Chat history
        while True:
            client.print_chain_status()

            prompt = input("\nEnter prompt: ")

            logger.info(f"Initiating text generation...")
            # full_prompt = conversation + "\n" + prompt
            full_prompt = prompt
            token_generator = client.generate(full_prompt, max_new_tokens=250, stream=True)

            if not token_generator:
                input("Press Enter to continue...")
                continue

            print(f"\n---------Response---------")
            response_text = ""
            for token in token_generator:
                response_text += token
                print(token, end="", flush=True)

            conversation += f"\nUser: {prompt}\nAssistant: {response_text}"
            print("\n")

            # Layer Reallocation
            client.trigger_reallocation()

    except KeyboardInterrupt:
        print("\nExiting...")  # Ctrl+C


if __name__ == "__main__":
    main()
