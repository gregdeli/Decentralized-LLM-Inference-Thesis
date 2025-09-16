from litgpt import LLM
import time

load_start = time.perf_counter()
llm = LLM.load("E:\GitHub\Decentralized-LLM-Inference-Thesis\models\litgpt_llama_3.2_1b\meta-llama\Llama-3.2-1B")
elapsed = time.perf_counter() - load_start
print(f"Model loading time: {elapsed:.2f} seconds")

gen_start = time.perf_counter()
text = llm.generate("The capital of France is", max_new_tokens=2, temperature=0.0, top_p=0.0)
elapsed = time.perf_counter() - gen_start
print(f"Total generation time: {elapsed:.2f} seconds")
print(text)

text = llm.generate("The tallest mountain in the world is", max_new_tokens=2, temperature=0.0)
print(text)

# Time: 00:58 mins
# from pprint import pprint

# text, bench_d = llm.benchmark(prompt="The capital of France is", max_new_tokens=10, temperature=0.0, top_p=0.0)
# print(text)
# pprint(bench_d)
