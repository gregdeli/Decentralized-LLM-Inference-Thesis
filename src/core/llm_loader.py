import json
import time
from typing import Dict, Any, Union, List, Optional, Tuple
from pathlib import Path
import torch
from safetensors.torch import load_file

from core.model import Llama3
from core.utils import remove_model_prefix

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
        is_client: bool = True,
        num_layers: int = None,
        layers_loaded: Tuple[int, int] = None,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.kv_cache_initialized = kv_cache_initialized

        self.is_client = is_client
        self.num_layers = num_layers
        self.layers_loaded = layers_loaded

    """
    High-level API for loading a Llama 3.2 model and generating text.
    
    It support dynamic transformer layer loading and split inference, by setting the is_client and num_layers attributs.
    The LLM object saves and publishes the transforemer layers that are loading in the current node's model 
    so that other nodes from which layer index to start loading.
    """

    @classmethod
    def load(
        cls,
        checkpoint_dir: Path,
        is_client: bool = True,
        num_layers: int = None,
        layers_start_idx: int = 0,
        time_it: bool = False,
    ) -> "LLM":
        if time_it:
            start_time = time.perf_counter()

        config_path = checkpoint_dir / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        if num_layers is None:
            num_layers = config["num_hidden_layers"]

        torch.set_float32_matmul_precision("high")

        # tokenizer = Tokenizer(checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

        model = Llama3(config, num_layers, is_client, layers_start_idx)
        model.eval()

        # Setup preprocessor
        preprocessor = Preprocessor(tokenizer, device="cpu")

        # Load weigths form the safetensors file
        weights_path = checkpoint_dir / "model.safetensors"
        state_dict = load_file(weights_path, device="cpu")
        state_dict = remove_model_prefix(state_dict)
        model.load_state_dict(state_dict, strict=False)

        # Save the transformer layers loaded in this node
        layers_end_idx = layers_start_idx + (num_layers - 1)
        layers_loaded = (layers_start_idx, layers_end_idx)

        if time_it:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"Model loading time: {elapsed_time: .2f} seconds")

        return cls(
            model=model,
            preprocessor=preprocessor,
            config=config,
            checkpoint_dir=checkpoint_dir,
            kv_cache_initialized=False,
            is_client=is_client,
            num_layers=num_layers,
            layers_loaded=layers_loaded,
        )

    # @torch.inference_mode()
    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        # sys_prompt: Optional[str] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        # top_p: float = 1.0,
        # return_as_token_ids: bool = False,
        stream: bool = False,
        time_it: bool = False,
    ) -> Union[str, List[str], iter]:

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
            if time_it:
                start = time.perf_counter()
            # Na allaksw to batch_size otan kanw batched inference
            self.model.set_kv_cache(batch_size=1, max_seq_length=max_returned_tokens, device=device)
            self.kv_cache_initialized = True
            if time_it:
                elapsed = time.perf_counter() - start
                print(f"KV cache initialization time: {elapsed:.5f} seconds")

        if stream:
            return self._generate_stream(input_ids, max_new_tokens, temperature, top_k)

        # If not streaming the output
        decoded_text = self._generate_fn(prompt_length, input_ids, max_new_tokens, temperature, top_k, time_it)

        return decoded_text

    @torch.no_grad()
    def _generate_fn(
        self,
        prompt_length: int,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        time_it: bool = False,
    ):
        if time_it:
            gen_start = time.perf_counter()

        generated_ids = []
        input = input_ids
        input_pos = None

        for _ in range(max_new_tokens):
            logits = self.model(input, input_pos=input_pos)
            logits = logits[:, -1, :]

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            if temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Stop if the end-of-sequence token is generated
            if next_token.item() == self.preprocessor.tokenizer.eos_token_id:
                break

            generated_ids.append(next_token)
            input = next_token
            current_pos = prompt_length + len(generated_ids)
            input_pos = torch.tensor([current_pos], device=self.preprocessor.device)

        all_generated_ids = torch.cat(generated_ids, dim=1)
        if time_it:
            elapsed = time.perf_counter() - gen_start
            print(f"Total generation time: {elapsed:.2f} seconds")
        return self.preprocessor.decode(all_generated_ids)

    @torch.no_grad()
    def _generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        """A generator function that yields decoded string chunks."""
        prompt_length = input_ids.size(1)
        input = input_ids
        input_pos = None

        for i in range(max_new_tokens):
            logits = self.model(input, input_pos=input_pos)
            logits = logits[:, -1, :]

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            if temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Stop if the end-of-sequence token is generated
            if next_token.item() == self.preprocessor.tokenizer.eos_token_id:
                break

            # Decode and yield the new token
            decoded_token = self.preprocessor.decode(next_token)
            yield decoded_token

            input = next_token
            # The position is `prompt_length` + tokens generated so far (which is i)
            current_pos = prompt_length + (i + 1)
            input_pos = torch.tensor([current_pos], device=self.preprocessor.device)


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
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        return inputs["input_ids"]

    def decode(self, output: torch.Tensor) -> str:
        # return self.tokenizer.decode(token_ids)
        decoded_texts = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return decoded_texts[0]


if __name__ == "__main__":
    model_path = Path(r"E:\GitHub\Decentralized-LLM-Inference-Thesis\models\Llama-3.2-1B")
    # llm = LLM.load(model_path, time_it=True)

    # prompt = "The capital of France is"
    # text = llm.generate(prompt, max_new_tokens=20, temperature=0.0, time_it=True)
    # print(prompt + text)

    # Streaming
    # prompt = "The Computer Enginnering and Informatics Department at the University of Patras is"
    # generator = llm.generate(prompt, max_new_tokens=100, temperature=0.0, stream=True)

    # print(prompt, end="", flush=True)
    # for e in generator:
    #     print(e, end="", flush=True)

    # Split inference Test
    # client = LLM.load(model_path, is_client=True, num_layers=8)

    # server_layers_start_idx = client.layers_loaded[1] + 1
    # server = LLM.load(model_path, is_client=False, num_layers=8, layers_start_idx=server_layers_start_idx)
