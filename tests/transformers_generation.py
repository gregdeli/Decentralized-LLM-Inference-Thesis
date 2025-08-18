# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("E:\GitHub\Decentralized-LLM-Inference-Thesis\models\Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("E:\GitHub\Decentralized-LLM-Inference-Thesis\models\Llama-3.2-1B")
