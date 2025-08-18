import json
from typing import Dict, Any, Union, Literal
from pathlib import Path
import torch
from safetensors.torch import load_file

from model import Llama3
from utils import remove_model_prefix

from litgpt.tokenizer import Tokenizer


class LLM:
    def __init__(
        self,
        model: Llama3,
        preprocessor=None,
        config: Dict[str, Any] = None,
        checkpoint_dir: Path = None,
        kv_cache_initialized: bool = False,
        fixed_kv_cache_size: Union[int, Literal["max_model_supported"], None] = None,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.kv_cache_initialized = kv_cache_initialized
        self.fixed_kv_cache_size = fixed_kv_cache_size

    """High-level API for loading a Llama 3.2 model and generating text"""

    @classmethod
    def load(
        cls,
        checkpoint_dir: Path,
    ) -> "LLM":
        config_path = checkpoint_dir / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        tokenizer = Tokenizer(checkpoint_dir)

        model = Llama3(config)
        model.eval()

        # Setup preprocessor
        preprocessor = Preprocessor(tokenizer, device="cpu")

        # Load weigths form the safetensors file
        weights_path = checkpoint_dir / "model.safetensors"
        state_dict = load_file(weights_path, device="cpu")
        state_dict = remove_model_prefix(state_dict)
        model.load_state_dict(state_dict, strict=False)

        return cls(
            model=model,
            preprocessor=preprocessor,
            config=config,
            checkpoint_dir=checkpoint_dir,
            kv_cache_initialized=False,
            fixed_kv_cache_size=False,
        )


class Preprocessor:
    """
    Preprocessor class for tokenization and de-tokenization.
    """

    def __init__(self, tokenizer: Tokenizer, device: str = "cpu") -> None:
        self.tokenizer = tokenizer
        self.device = device

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text, device=self.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids)
