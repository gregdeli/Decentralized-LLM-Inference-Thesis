"""Main client application logic"""

import grpc
import os
import torch
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple
import time
import logging

from core.llm_loader import LLM
from core.remote import nodeservice_pb2, nodeservice_pb2_grpc
from core.remote.serialization import tensor_to_request, response_to_tensor
from core.p2p.dht_manager import DHTManager
from core.p2p.chain_manager import ChainManager

logger = logging.getLogger(__name__)


class Client:
    def __init__(
        self,
        model_path: Path,
        host_maddrs: List[str] = "/ip4/0.0.0.0/tcp/4001",
        initial_peers: List[str] = None,
        time_it: bool = False,
    ) -> None:
        self.dht = DHTManager(host_maddrs=[host_maddrs], initial_peers=initial_peers)
        self.dht.start()
        self.chain = ChainManager(self.dht)

        self.llm = LLM.load(model_path, is_client=True, num_layers=0, time_it=time_it)
        self.model = self.llm.model

        head_info = self.chain.get_head_server_info(num_total_layers=self.llm.config["num_hidden_layers"])
        if not head_info:
            raise RuntimeError("Client could not find the head server.")
            
        # logger.info(f"Client successfully found head server. Address: {head_info['address']}")

        head_server_addr = head_info["address"]
        channel = grpc.insecure_channel(head_server_addr)
        self.head_server_stub = nodeservice_pb2_grpc.NodeServiceStub(channel)

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 50,
        temperature: float = 0.5,
        top_p: float = 0.9,
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

        generated_ids = []
        input_tensor = input_ids
        input_pos = None
        seq_length = prompt_length
        for i in range(max_new_tokens):
            logger.info(f"Generating token {i + 1}/{max_new_tokens}") # Debugging

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
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Stop if the end-of-sequence token is generated
            if next_token.item() == self.llm.preprocessor.tokenizer.eos_token_id:
                break

            generated_ids.append(next_token)
            input_tensor = next_token
            current_pos = prompt_length + len(generated_ids)
            input_pos = torch.tensor([current_pos], device=self.llm.preprocessor.device)
            seq_length = 1

        all_generated_ids = torch.cat(generated_ids, dim=1)

        return self.llm.preprocessor.decode(all_generated_ids)
