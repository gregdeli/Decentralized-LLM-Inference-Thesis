import os
from pathlib import Path
from client.client import Client

if __name__ == "__main__":
    model_path_str = os.getenv("MODEL_PATH")
    model_path = Path(model_path_str)
    head_server_addr = os.getenv("HEAD_SERVER_ADDR")

    client = Client(model_path=model_path, head_server_addr=head_server_addr)

    prompt = "The capital of France is"
    text = client.generate(prompt, max_new_tokens=20)
    print(prompt + text)
