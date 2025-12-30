import json
from core.utils import calculate_transformer_params

# MODEL_PATH = "/home/greg_deli/Desktop/Decentralized-LLM-Inference-Thesis/models/Llama-3.2-3B-Instruct"
MODEL_PATH = "/home/greg/Desktop/Decentralized-LLM-Inference-Thesis/models/Llama-3.2-1B-Instruct"

param_count = calculate_transformer_params(MODEL_PATH)

with open(f"{MODEL_PATH}/config.json", "r") as f:
    config = json.load(f)

config["total_transformer_layer_params"] = param_count


with open(f"{MODEL_PATH}/config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"Updated {MODEL_PATH}/config.json with total_transformer_layer_params: {param_count}")
