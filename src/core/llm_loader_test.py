from llm_loader import LLM
from pathlib import Path

model_path = Path(r"E:\GitHub\Decentralized-LLM-Inference-Thesis\models\Llama-3.2-1B")
llm = LLM.load(model_path)

prompt = "The capital of France is"
texts = llm.generate(prompt, max_new_tokens=50, temperature=0)
print(prompt + texts[0])

# Time: 2:01 mins
