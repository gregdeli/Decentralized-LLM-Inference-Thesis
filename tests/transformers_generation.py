# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("E:\GitHub\Decentralized-LLM-Inference-Thesis\models\Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("E:\GitHub\Decentralized-LLM-Inference-Thesis\models\Llama-3.2-1B")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=2)
output = outputs[0]
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)

# Time: 1:00 mins
