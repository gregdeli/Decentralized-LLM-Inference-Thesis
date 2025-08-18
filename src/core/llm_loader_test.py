from llm_loader import LLM
from pathlib import Path

model_path = Path(r"E:\GitHub\Decentralized-LLM-Inference-Thesis\models\Llama-3.2-1B")
llm = LLM.load(model_path)

print("hello")
