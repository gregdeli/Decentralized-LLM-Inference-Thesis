from litgpt import LLM

llm = LLM.load("E:\GitHub\Decentralized-LLM-Inference-Thesis\models\litgpt_llama_3.2_1b\meta-llama\Llama-3.2-1B")

# Print the full generated text
text = llm.generate("The capital of France is", max_new_tokens=2, temperature=0.0)
print(text)

# Time: 00:58 mins

# Stream the response one token at a time
# result = llm.generate(
#     "The capital of France is",
#     stream=True,
# )
# for e in result:
#     print(e, end="", flush=True)
