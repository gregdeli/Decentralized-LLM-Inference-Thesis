import json
from typing import Dict, Any, Union, Literal, Optional
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
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.kv_cache_initialized = kv_cache_initialized

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
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        sys_prompt: Optional[str] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        return_as_token_ids: bool = False,
    ) -> Union[str, torch.Tensor]:

        input_ids = self.preprocessor.encode(prompt)
        prompt_length = input_ids.size(0)
        max_returned_tokens = prompt_length + max_new_tokens

        if max_returned_tokens > self.model.max_seq_length:
            raise ValueError(
                f"The combined prompt and max_new_tokens length ({max_returned_tokens}) exceeds "
                f"the model's maximum sequence length of {self.model.max_seq_length}."
            )

        if not self.kv_cache_initialized:
            device = self.preprocessor.device
            # Always allocate the cache to the maximum possible size.
            # This could change in the future if the kv cache takes up to much memory
            self.model.set_kv_cache(batch_size=1, max_seq_length=self.model.max_seq_length, device=device)
            self.kv_cache_initialized = True

        # Generate tokens
        self.model.eval()
        # Sunexizw edw + implement ta forward methods


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
