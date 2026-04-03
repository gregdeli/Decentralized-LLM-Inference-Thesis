import os
from dotenv import load_dotenv
from pathlib import Path
import logging

from client.client import Client
from core.remote.utils import get_bootstrap_peer_address, get_ip_address, discover_bootstrap_node_address

logger = logging.getLogger(__name__)

# To not show netlinkrib errors on moto
os.environ["GOLOG_LOG_LEVEL"] = "fatal"

GPRC_PORT = 5001

def main():
    load_dotenv()
    model_path_str = os.getenv("MODEL_PATH")

    my_ip = os.getenv("IP")
    if not my_ip:
        my_ip = get_ip_address()
    host_maddrs = f"/ip4/{my_ip}/tcp/0"

    bootstrap_addr = os.getenv("BOOTSTRAP_NODE_ADDR")

    if not bootstrap_addr:
        bootstrap_addr = discover_bootstrap_node_address()

    if not bootstrap_addr:
        logger.error("Failed to find bootstrap node.")
        return

    bootstrap_peer_addr = get_bootstrap_peer_address(bootstrap_addr, attempts=5)
    initial_peers = [bootstrap_peer_addr] if bootstrap_peer_addr else None

    # Initialize the Client Node
    grpc_addr = grpc_addr = f"{my_ip}:{GPRC_PORT}"
    client = Client(
        model_path=Path(model_path_str),
        host_maddrs=[host_maddrs],
        initial_peers=initial_peers,
        grpc_addr = grpc_addr
    )
    logger.info("Client initialized.")

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

            # Layer Reallocation
            #client.trigger_reallocation()

    except KeyboardInterrupt:
        print("\nExiting...")  # Ctrl+C


if __name__ == "__main__":
    main()
