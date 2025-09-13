"""Main client application logic"""

import grpc
import os
import torch
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple
import time

from core.llm_loader import LLM
from core.remote import inference_pb2, inference_pb2_grpc
from core.remote.serialization import tensor_to_request, response_to_tensor

# from core.dht import DHT


class Client:
    def __init__(
        self,
        model_path: Path,
        head_server_addr: str,
        time_it: bool = False,
    ) -> None:
        self.llm = LLM.load(model_path, is_client=True, num_layers=0, time_it=time_it)
        self.model = self.llm.model

        channel = grpc.insecure_channel(head_server_addr)
        self.head_server_stub = inference_pb2_grpc.InferenceStub(channel)

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        stream: bool = False,
        time_it: bool = False,
    ) -> Union[str, List[str], iter]:

        input_ids = self.llm.preprocessor.encode(prompt)
        prompt_length = input_ids.size(1)
        max_returned_tokens = prompt_length + max_new_tokens

        if max_returned_tokens > self.model.max_seq_length:
            raise ValueError(
                f"The combined prompt and max_new_tokens length ({max_returned_tokens}) exceeds "
                f"the model's maximum sequence length of {self.model.max_seq_length}."
            )

        if not self.llm.kv_cache_initialized:
            device = self.llm.preprocessor.device
            # Na allaksw to batch_size otan kanw batched inference
            self.model.set_kv_cache(batch_size=1, max_seq_length=max_returned_tokens, device=device)
            self.llm.kv_cache_initialized = True

        generated_ids = []
        input_tensor = input_ids
        input_pos = None
        seq_length = prompt_length
        for _ in range(max_new_tokens):
            x = self.model.forward_client_initial(input_tensor, input_pos=input_pos)

            # Call the remote server chain
            request = tensor_to_request(
                x, max_returned_tokens=max_returned_tokens, seq_length=seq_length, input_pos=input_pos.item() if input_pos is not None else None
            )

            response = self.head_server_stub.RunLayers(request)
            x = response_to_tensor(response)

            # Run clients final layers
            logits = self.model.forward_client_final(x)

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
            if next_token.item() == self.llm.preprocessor.tokenizer.eos_token_id:
                break

            generated_ids.append(next_token)
            input = next_token
            current_pos = prompt_length + len(generated_ids)
            input_pos = torch.tensor([current_pos], device=self.llm.preprocessor.device)
            seq_length = 1

        all_generated_ids = torch.cat(generated_ids, dim=1)

        return self.llm.preprocessor.decode(all_generated_ids)
