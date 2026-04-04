from huggingface_hub import snapshot_download
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
model_dest = str(project_root / "models/Llama-3.2-1B-Instruct")

snapshot_download(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    token="",
    local_dir=model_dest,
    ignore_patterns="original/*",
)
