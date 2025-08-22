import json
from typing import Dict, Any, Union, Literal, Optional
from pathlib import Path
import torch
from safetensors.torch import load_file

from model import Llama3
from utils import remove_model_prefix

# from litgpt.tokenizer import Tokenizer
from transformers import AutoTokenizer


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

        # tokenizer = Tokenizer(checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

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
        # sys_prompt: Optional[str] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        # top_p: float = 1.0,
        # return_as_token_ids: bool = False,
    ) -> Union[str, torch.Tensor]:

        input_ids = self.preprocessor.encode(prompt)
        prompt_length = input_ids.size(1)

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
            # Na allaksw to batch_size otan kanw batched inference
            self.model.set_kv_cache(batch_size=1, max_seq_length=self.model.max_seq_length, device=device)
            self.kv_cache_initialized = True

        # Auto-regressive generation loop
        input = input_ids
        input_pos = None
        generated_ids = []
        for _ in range(max_new_tokens):
            logits = self.model(input, input_pos=input_pos)
            logits = logits[:, -1, :]  # Last token logits

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set all logits not in the top-k to -inf
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Apply temperature scaling
            if temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy sampling
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Stop if end-of-sequence token is generated
            if next_token.item() == self.preprocessor.tokenizer.eos_token:
                break

            generated_ids.append(next_token)
            input = next_token

            current_pos = prompt_length + len(generated_ids)
            input_pos = torch.tensor([current_pos], device=self.preprocessor.device)

        all_generated_ids = torch.cat(generated_ids, dim=1)
        return self.preprocessor.decode(all_generated_ids)


class Preprocessor:
    """
    Preprocessor class for tokenization and de-tokenization.
    """

    def __init__(self, tokenizer: AutoTokenizer, device: str = "cpu") -> None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        self.device = device

    def encode(self, text: str) -> torch.Tensor:
        # return self.tokenizer.encode(text, device=self.device)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        return inputs["input_ids"]

    def decode(self, outputs: torch.Tensor) -> str:
        # return self.tokenizer.decode(token_ids)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    model_path = Path(r"E:\GitHub\Decentralized-LLM-Inference-Thesis\models\Llama-3.2-1B")
    llm = LLM.load(model_path)

    prompt = "The capital of France is"
    texts = llm.generate(prompt, max_new_tokens=10, temperature=0.0)
    print(prompt + texts[0])

    # prompt = "The meaning of life is"
    # texts = llm.generate(prompt, max_new_tokens=10, temperature=0.0)
    # print(prompt + texts[0])
