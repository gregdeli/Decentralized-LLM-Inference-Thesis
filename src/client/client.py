"""Main client application logic"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
import time

import grpc
import torch

from core.llm_loader import LLM
from core.remote import nodeservice_pb2, nodeservice_pb2_grpc
from core.remote.serialization import *
from core.p2p.dht_manager import DHTManager
from core.p2p.chain_manager import ChainManager

logger = logging.getLogger(__name__)


class Client:
    def __init__(
        self,
        model_path: Path,
        host_maddrs: List[str] = ["/ip4/0.0.0.0/tcp/4001"],
        initial_peers: List[str] = None,
        time_it: bool = False,
    ) -> None:
        self.dht = DHTManager(host_maddrs=host_maddrs, initial_peers=initial_peers)
        self.dht.start()
        self.chain = ChainManager(self.dht)

        self.llm = LLM.load(model_path, is_client=True, time_it=time_it)
        self.model = self.llm.model

        head_info = self.chain.get_head_server_info()
        if not head_info:
            raise RuntimeError("Client could not find the head server.")

        logger.info(f"Client successfully found head server. Address: {head_info['address']}")

        head_server_addr = head_info["address"]

        max_msg_size = 100 * 1024 * 1024
        channel = grpc.insecure_channel(
            head_server_addr,
            options=[
                ("grpc.max_send_message_length", max_msg_size),
                ("grpc.max_receive_message_length", max_msg_size),
            ],
        )
        self.head_server_stub = nodeservice_pb2_grpc.NodeServiceStub(channel)

        self.total_rate = 0.0
        self.last_inference_stats = {"latency": 0.0, "throughput": 0.0}

    def print_chain_status(self):
        self.chain.print_chain_status()

    def trigger_reallocation(self):
        """
        Triggers the layer reallocation process starting from the HEAD.
        """
        if self.total_rate > 0:
            logger.info(f"Triggering reallocation with Total Rate: {self.total_rate:.2f} layers/sec...")

            request = nodeservice_pb2.ReallocateRequest(total_rate=self.total_rate, start_layer_index=0)

            try:
                self.head_server_stub.Reallocate(request)
                logger.info("Reallocation triggered successfully...")
            except grpc.RpcError as e:
                logger.error(f"Failed to trigger reallocation: {e}")

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 50,
        temperature: float = 0.6,
        top_p: float = 0.9,
        stream: bool = False,
        time_it: bool = False,
    ) -> Union[str, Iterator[str]]:

        # Determine if all the layers have been loaded on the server chain
        all_layers_loaded = self.chain.get_all_layers_loaded()
        if not all_layers_loaded:
            warning = "Not all model layers have been loaded on the server chain. Cannot initiate the generation task."
            logger.warning(warning)
            return warning

        prompt = self.llm.apply_chat_template(prompt)
        input_ids = self.llm.preprocessor.encode(prompt)

        prompt_length = input_ids.size(1)
        max_returned_tokens = prompt_length + max_new_tokens

        if max_returned_tokens > self.model.max_seq_length:
            raise ValueError(
                f"The combined prompt and max_new_tokens length ({max_returned_tokens}) exceeds "
                f"the model's maximum sequence length of {self.model.max_seq_length}."
            )

        if stream:
            return self._generate_stream(prompt_length, input_ids, max_new_tokens, max_returned_tokens, temperature, top_p, time_it)

        # If not streaming the output
        decoded_text = self._generate_fn(prompt_length, input_ids, max_new_tokens, max_returned_tokens, temperature, top_p, time_it)
        return decoded_text

    @torch.no_grad()
    def _generate_fn(
        self,
        prompt_length: int,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        max_returned_tokens: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        time_it: bool = False,
    ) -> str:
        generated_ids = []
        input_tensor = input_ids
        input_pos = None
        seq_length = prompt_length
        for _ in range(max_new_tokens):
            # logger.info(f"Generating token {i + 1}/{max_new_tokens}")  # Debugging
            x = self.model.forward_client_initial(input_tensor, input_pos=input_pos)

            # Call the remote server chain
            request = tensor_to_request(
                x, max_returned_tokens=max_returned_tokens, seq_length=seq_length, input_pos=input_pos.item() if input_pos is not None else None
            )

            response = self.head_server_stub.RunLayers(request)

            if response.HasField("error_message"):
                logger.error(f"Server-side failure: {response.error_message}")
                logger.error("Aborting generation task. Please try again.")
                return

            # Capture TOTAL RATE
            if response.total_rate > 0:
                self.total_rate = response.total_rate

            x = message_to_tensor(response)

            # Run clients final layers
            logits = self.model.forward_client_final(x)

            # Sample the next token
            next_token = self.llm.sample_logits(logits, temperature, top_p)

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

    @torch.no_grad()
    def _generate_stream(
        self,
        prompt_length: int,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        max_returned_tokens: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        time_it: bool = False,
    ) -> Iterator[str]:
        input_tensor = input_ids
        input_pos = None
        seq_length = prompt_length

        start_time = time.perf_counter()
        tokens_generated = 0
        for i in range(max_new_tokens):
            # logger.info(f"Generating token {i + 1}/{max_new_tokens}")  # Debugging
            x = self.model.forward_client_initial(input_tensor, input_pos=input_pos)

            # Call the remote server chain
            request = tensor_to_request(
                x, max_returned_tokens=max_returned_tokens, seq_length=seq_length, input_pos=input_pos.item() if input_pos is not None else None
            )

            response = self.head_server_stub.RunLayers(request)

            if response.HasField("error_message"):
                logger.error(f"Server-side failure: {response.error_message}")
                logger.error("Aborting generation task. Please try again.")
                return

            # Capture TOTAL RATE
            if response.total_rate > 0:
                self.total_rate = response.total_rate

            x = message_to_tensor(response)

            # Run clients final layers
            logits = self.model.forward_client_final(x)

            # Sample the next token
            next_token = self.llm.sample_logits(logits, temperature, top_p)

            # Stop if the end-of-sequence token is generated
            if next_token.item() == self.llm.preprocessor.tokenizer.eos_token_id:
                break

            # Decode and yield the new token
            decoded_token = self.llm.preprocessor.decode(next_token)
            tokens_generated += 1

            yield decoded_token

            input_tensor = next_token
            current_pos = prompt_length + (i + 1)
            input_pos = torch.tensor([current_pos], device=self.llm.preprocessor.device)
            seq_length = 1

        elapsed_time = time.perf_counter() - start_time
        throughput = tokens_generated / elapsed_time if elapsed_time > 0 else 0

        self.last_inference_stats = {"latency": elapsed_time, "throughput": throughput}
