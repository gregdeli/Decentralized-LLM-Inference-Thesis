from pathlib import Path
import json

from core.llm_loader import LLM

# model_path = Path(r"/home/greg_deli/Desktop/Decentralized-LLM-Inference-Thesis/models/Llama-3.2-3B-Instruct")
model_path = Path(r"/home/greg_deli/Desktop/Decentralized-LLM-Inference-Thesis/models/Llama-3.2-1B-Instruct")

with open(f"{model_path}/config.json", "r") as f:
    config = json.load(f)

num_layers = config.get("num_hidden_layers")

llm = LLM.load(model_path, layers_to_load=(0, num_layers - 1), time_it=True)

# No Stream
# prompt = "What is the capital of France?"
# prompt = "What sport did Michael Jordan play? Give me a single word answer."
# text = llm.generate(prompt, max_new_tokens=50, stream=True)
# print(f"--------Prompt--------\n{prompt}")
# print(f"--------Response--------\n{text}")

# Stream Output
try:
    while True:
        prompt = input("\nEnter prompt (or Ctrl+C to quit): ")

        print(f"\n-----Response-----\n", end="", flush=True)

        token_generator = llm.generate(prompt, max_new_tokens=250, stream=True)

        # Iterate over the generator and print each token as it arrives
        try:
            for token in token_generator:
                print(token, end="", flush=True)
        except KeyboardInterrupt:
            print("\nStopping text generation...")

        print("\n")

except KeyboardInterrupt:
    print("\nExiting...")
