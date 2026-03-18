from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    token="",
    local_dir="/home/greg/Desktop/Decentralized-LLM-Inference-Thesis/models/Llama-3.2-1B-Instruct",
    ignore_patterns="original/*",
)
