import os
from pathlib import Path
import time
import grpc

from client.client import Client
from core.remote import nodeservice_pb2, nodeservice_pb2_grpc

def main():
    model_path_str = os.getenv("MODEL_PATH")
    model_path = Path(model_path_str)
    
    host_maddrs = os.getenv("HOST_MADDRS")
    bootstrap_node_addr_str = os.getenv("BOOTSTRAP_NODE_ADDR")

    connected = False
    for attempt in range(5):
        try:
            print(f"Attempting to discover bootstrap peer at {bootstrap_node_addr_str} (Attempt {attempt + 1})...")
            with grpc.insecure_channel(bootstrap_node_addr_str) as channel:
                stub = nodeservice_pb2_grpc.NodeServiceStub(channel)
                response = stub.GetPeerMultiaddr(nodeservice_pb2.Empty())
                initial_peers = [response.multiaddr]
            print(f"Successfully discovered bootstrap peer: {initial_peers[0]}")
            connected = True
            break
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                print("Bootstrap node not ready yet, retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"An unexpected gRPC error occurred: {e}")
                break 
    
    if not connected:
        print("FATAL: Could not connect to bootstrap node after several attempts.")
        return

    client = Client(
        model_path=model_path,
        host_maddrs=host_maddrs,
        initial_peers=initial_peers,
    )

    prompt = "The capital of France is"
    text = client.generate(prompt, max_new_tokens=2)
    print(prompt + text)

    prompt = "The capital of Greece is"
    text = client.generate(prompt, max_new_tokens=15)
    print(prompt + text)

if __name__ == "__main__":
    main()    
