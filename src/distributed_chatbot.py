"""
usage: distributed_chatbot.py [-h] [--prompt PROMPT] [--max_new_tokens MAX_NEW_TOKENS] [--stream]
"""

import os
import argparse
from dotenv import load_dotenv
from pathlib import Path
import logging

from client.client import Client
from core.remote.utils import (
    get_bootstrap_peer_address, 
    get_ip_address, 
    discover_bootstrap_node_address,
    get_free_port
)

logger = logging.getLogger(__name__)

# To not show netlinkrib errors on moto
os.environ["GOLOG_LOG_LEVEL"] = "fatal"

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Distributed LLM Inference Chatbot")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt for the LLM. If provided, the script runs once and exits.")
    parser.add_argument("--max_new_tokens", type=int, default=250, help="Maximum number of new tokens to generate.")
    parser.add_argument("--stream", action="store_true", help="Enable token streaming output.")
    args = parser.parse_args()

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
    grpc_port = os.getenv("GRPC_PORT")
    grpc_port = grpc_port if grpc_port else get_free_port()
    grpc_addr = grpc_addr = f"{my_ip}:{grpc_port}"

    # Set log level
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.getLogger().setLevel(getattr(logging, log_level_str))

    quantize_flag_str = os.getenv("QUANTIZE", "1")
    quantize_flag = quantize_flag_str.lower() in ("1", "true", "yes")

    # Initialize the Client Node
    client = Client(
        model_path=Path(model_path_str),
        host_maddrs=[host_maddrs],
        initial_peers=initial_peers,
        grpc_addr = grpc_addr,
        quantize_flag=quantize_flag
    )
    logger.info("Client initialized.")

    try:
        # Single-shot execution if a prompt is passed via CLI
        if args.prompt:
            logger.info("Initiating text generation...")
            token_generator = client.generate(
                args.prompt, 
                max_new_tokens=args.max_new_tokens, 
                stream=args.stream
            )

            if not token_generator:
                print("Error!")
                return

            print("\n---------Response---------")

            if isinstance(token_generator, str):
                print(token_generator)

            else:
                try:
                    for token in token_generator:
                        print(token, end="", flush=True)
                except KeyboardInterrupt:
                    print("\nStopping text generation...")

                stats = client.last_inference_stats
                latency = stats.get("latency")
                throughput = stats.get("throughput")
                print(f"\n\nGeneration Time: {latency:.2f}s")
                print(f"Throughput: {throughput:.2f} tokens/sec\n")
            return

        while True:
            # client.print_chain_status()

            prompt = input("\nEnter prompt: ")

            logger.info(f"Initiating text generation...")

            token_generator = client.generate(prompt, max_new_tokens=args.max_new_tokens, stream=args.stream)

            if not token_generator:
                input("Press Enter to continue...")
                continue

            print(f"\n---------Response---------")
            
            # Buffered generation
            if isinstance(token_generator, str):
                print(token_generator)

            # Streaming Generation
            else:
                try:
                    for token in token_generator:
                        print(token, end="", flush=True)
                except KeyboardInterrupt:
                    print("\nStopping text generation...")

            # Stats 
            stats = client.last_inference_stats
            latency = stats.get("latency")
            throughput = stats.get("throughput")
            print(f"\n\nGeneration Time: {latency:.2f}s")
            print(f"Throughput: {throughput:.2f} tokens/sec")
            print("\n")

            # Layer Reallocation
            #client.trigger_reallocation()

    except KeyboardInterrupt:
        print("\nExiting...")  # Ctrl+C


if __name__ == "__main__":
    main()
