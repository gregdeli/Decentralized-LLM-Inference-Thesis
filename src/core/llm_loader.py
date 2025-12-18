import json
import logging
import time
from typing import Dict, Any, Union, List, Optional, Tuple, Iterator
from pathlib import Path
import torch
import gc
from safetensors.torch import load_file

from core.model import Llama3
from core.utils import remove_model_prefix, is_instruct_model, get_relevant_safetensor_files

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class LLM:
    def __init__(
        self,
        model: Llama3,
        preprocessor=None,
        config: Dict[str, Any] = None,
        is_instruct_model: bool = False,
        model_path: Path = None,
        kv_cache_initialized: bool = False,
        is_client: bool = True,
        layers_loaded: Tuple[int, int] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.is_instruct_model = is_instruct_model
        self.model_path = model_path
        self.kv_cache_initialized = kv_cache_initialized
        self.prev_generated_seq_length = 0

        self.is_client = is_client
        self.layers_loaded = layers_loaded
        self.device = device

    """
    High-level API for loading a Llama 3.2 model and generating text.
    
    It support dynamic transformer layer loading and split inference, by setting the is_client and num_layers attributs.
    The LLM object saves and publishes the transformer layers that are loaded in the current node's model 
    so that other nodes know from which layer index to start loading.
    """

    @classmethod
    def load(
        cls,
        model_path: Path,
        is_client: bool = True,
        layers_to_load: Tuple[int, int] = None,
        time_it: bool = False,
    ) -> "LLM":
        if time_it:
            start_time = time.perf_counter()

        # Check for CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "cpu"

        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # Check is the model is instruction tuned
        is_instruct = is_instruct_model(model_path=model_path)

        # tokenizer = Tokenizer(checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = Llama3(config, is_client, layers_to_load)
        model.eval()

        # Setup preprocessor
        preprocessor = Preprocessor(tokenizer, device=device)

        # Load weigths from the safetensors file or files
        index_path = model_path / "model.safetensors.index.json"
        single_file_path = model_path / "model.safetensors"

        files_to_load = []

        if index_path.exists():
            # Sharded Checkpoint
            with open(index_path, "r") as f:
                index_data = json.load(f)

            weight_map = index_data.get("weight_map")
            files_to_load = get_relevant_safetensor_files(weight_map)
            logger.info(f"Identified {len(files_to_load)} relevant checkpoint shards.")

        elif single_file_path.exists():
            # Single Checkpoint
            files_to_load = ["model.safetensors"]

        else:
            raise FileNotFoundError(f"No safetensors model found at {model_path}")

        logger.info("Loading state dict...")
        for filename in files_to_load:
            file_path = model_path / filename
            state_dict = load_file(file_path)
            state_dict = remove_model_prefix(state_dict)
            model.load_state_dict(state_dict, strict=False)

            # Explicitly free memory
            del state_dict
            gc.collect()

        logger.info(f"Moving model to {device}...")
        model.to(device)  # Move parameters to VRAM if gpu available
        logger.info(f"Model successfully moved to {device}...")

        if time_it:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"Model loading time: {elapsed_time: .2f} seconds")

        return cls(
            model=model,
            preprocessor=preprocessor,
            config=config,
            is_instruct_model=is_instruct,
            model_path=model_path,
            kv_cache_initialized=False,
            is_client=is_client,
            layers_loaded=layers_to_load,
            device=device,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        # sys_prompt: Optional[str] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.6,
        top_p: float = 0.9,
        stream: bool = False,
        time_it: bool = False,
    ) -> Union[str, Iterator[str]]:

        prompt = self.apply_chat_template(prompt)

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

        # Dynamically grow the kv cache size if necessary
        elif self.prev_generated_seq_length < max_returned_tokens:
            tmp_device = self.model.mask_cache.device
            self.model.clear_kv_cache()
            self.model.set_kv_cache(batch_size=1, max_seq_length=max_returned_tokens, device=tmp_device)

        self.prev_generated_seq_length = max_returned_tokens

        if stream:
            return self._generate_stream(input_ids, max_new_tokens, temperature, top_p)

        # If not streaming the output
        decoded_text = self._generate_fn(prompt_length, input_ids, max_new_tokens, temperature, top_p, time_it)

        return decoded_text

    @torch.no_grad()
    def _generate_fn(
        self,
        prompt_length: int,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        time_it: bool = False,
    ) -> str:
        if time_it:
            gen_start = time.perf_counter()

        generated_ids = []
        input = input_ids
        input_pos = None

        for _ in range(max_new_tokens):
            logits = self.model(input, input_pos=input_pos)

            next_token = self.sample_logits(logits, temperature, top_p)

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
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> Iterator[str]:
        """A generator function that yields decoded string chunks."""
        prompt_length = input_ids.size(1)
        input = input_ids
        input_pos = None

        for i in range(max_new_tokens):
            logits = self.model(input, input_pos=input_pos)

            next_token = self.sample_logits(logits, temperature, top_p)

            # Stop if the end-of-sequence token is generated
            if next_token.item() == self.preprocessor.tokenizer.eos_token_id:
                break

            # Decode and yield the new token
            decoded_token = self.preprocessor.decode(next_token)
            yield decoded_token

            input = next_token
            current_pos = prompt_length + (i + 1)
            input_pos = torch.tensor([current_pos], device=self.preprocessor.device)

    def apply_chat_template(self, prompt: str) -> str:
        """
        Applies the Llama 3.2 Instruct chat template to the prompt if the model used is instruction tuned.
        """
        default_system_prompt = "You are a helpful assistant"

        if self.is_instruct_model:
            return (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{default_system_prompt}<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            return prompt

    def sample_logits(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        """Applies temperature and top-p (nucleus) sampling to logits."""
        logits = logits[:, -1, :]

        if top_p > 0.0:
            # Sort logits and compute probabilities
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Find the indices to remove (those outside the nucleus)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first one that exceeds top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Create a mask to set the logits of tokens to remove to -inf
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float("Inf")

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        return next_token


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
    model_path = Path(r"/home/greg/Decentralized-LLM-Inference-Thesis/models/Llama-3.2-1B")
    llm = LLM.load(model_path)

    prompt = "The capital of France is"
    text = llm.generate(prompt, max_new_tokens=2)
    print(prompt + text)

    # prompt = "The meaning of life is"
    # text = llm.generate(prompt, max_new_tokens=6)
    # print(prompt + text)

    # prompt = "The tallest mountain in the world is"
    # text = llm.generate(prompt, max_new_tokens=2)
    # print(prompt + text)

    # Streaming
    # prompt = "The Computer Enginnering and Informatics Department at the University of Patras is"
    # generator = llm.generate(prompt, max_new_tokens=100, temperature=0.0, stream=True)

    # print(prompt, end="", flush=True)
    # for e in generator:
    #     print(e, end="", flush=True)
