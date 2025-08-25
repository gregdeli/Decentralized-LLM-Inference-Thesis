from pathlib import Path
from client.client import Client
from server.server import Server
from core.dht import DHT

"""
client:      prompt -> encode -> embed -> layers.0-n (optionally) ->
head_server: layers.n+1-m ->
tail_server: layers.m+1-15 ->
client:      final_norm -> lm_head 
client:      all_generated_tokens -> decode 
"""

model_path = Path(r"E:\GitHub\Decentralized-LLM-Inference-Thesis\models\Llama-3.2-1B")
client = Client(model_path, num_layers=5)  # layers.0-4

start_idx = start_idx = client.llm.layers_loaded[1] + 1
head_server = Server(model_path, num_layers=6, layers_start_idx=start_idx, is_head=True)  # layers.5-10

start_idx = head_server.llm.layers_loaded[1] + 1
tail_server = Server(model_path, num_layers=5, layers_start_idx=start_idx)  # layers.11-15

head_server.successor = tail_server


DHT.register_client(client)
DHT.register_servers([head_server, tail_server])

prompt = "The capital of France is"
text = client.generate(prompt, max_new_tokens=20)
print(prompt + text)
