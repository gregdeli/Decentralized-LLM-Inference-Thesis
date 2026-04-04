from huggingface_hub import snapshot_download
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
model_dest = str(project_root / "models/Llama-3.2-1B")

# Den to kanei load me litgpt ->
# FileNotFoundError: checkpoint_dir 'E:\\GitHub\\Decentralized-LLM-Inference-Thesis\\models\\Llama-3.2-1B' is missing the files:
# ['lit_model.pth', 'model_config.yaml'].
snapshot_download(
    repo_id="meta-llama/Llama-3.2-1B",
    token="",
    local_dir=model_dest,
)

# Download models with litgpt cli tool
# litgpt download --access_token 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 --checkpoint_dir E:\GitHub\Decentralized-LLM-Inference-Thesis\models\litgpt_llama_3.2_1b meta-llama/Llama-3.2-1B
