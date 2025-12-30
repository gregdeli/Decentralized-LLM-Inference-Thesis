from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    token="",
    local_dir="/home/greg_deli/Desktop/Decentralized-LLM-Inference-Thesis/models/Llama-3.2-3B-Instruct",
    ignore_patterns="original/*",
)
