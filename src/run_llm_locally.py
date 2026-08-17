from pathlib import Path
import json
import time
import logging

from core.llm_loader import LLM

logger = logging.getLogger(__name__)

# model_path = Path("/models/Llama-3.2-3B-Instruct")
model_path = Path("models/Llama-3.2-1B-Instruct")

with open(f"{model_path}/config.json", "r") as f:
    config = json.load(f)

num_layers = config.get("num_hidden_layers")

llm = LLM.load(model_path, load_initial_layer=True, layers_to_load=(0, num_layers - 1), load_output_layer=True, time_it=True)

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

        token_generator = llm.generate(prompt, max_new_tokens=1024, stream=True)

        # Iterate over the generator and print each token as it arrives
        try:
            tokens_generated = 0
            start_time = time.perf_counter()
            for token in token_generator:
                print(token, end="", flush=True)
                tokens_generated += 1
            elapsed_time = time.perf_counter() - start_time
        except KeyboardInterrupt:
            elapsed_time = time.perf_counter() - start_time
            print("\nStopping text generation...")

        throughput = tokens_generated / elapsed_time if elapsed_time > 0 else 0
        print("\n")
        print(f"Generation Time: {elapsed_time:.2f}s")
        print(f"Throughput: {throughput:.2f} tokens/sec")

except KeyboardInterrupt:
    print("\nExiting...")
