from pathlib import Path

from core.llm_loader import LLM

model_path = Path(r"/home/greg/Decentralized-LLM-Inference-Thesis/models/Llama-3.2-1B-Instruct")
llm = LLM.load(model_path, layers_to_load=(0, 15))

# prompt = "What is the capital of France?"
prompt = "What sport did Michael Jordan play? Give me a single word answer."
text = llm.generate(prompt, max_new_tokens=50)
print(f"-----Prompt-----\n{prompt}")
print(f"\n-----Response-----\n{text}")
