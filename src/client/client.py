"""Main client application logic"""

from pprint import pprint
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
import time
import threading
from concurrent import futures

import grpc
import torch

from client.servicer import ClientServicer
from core.llm_loader import LLM
from core.utils import can_load
from core.remote import nodeservice_pb2, nodeservice_pb2_grpc
from core.remote.utils import get_ip_address, create_grpc_channel
from core.remote.serialization import *
from core.p2p.dht_manager import DHTManager
from core.p2p.chain_manager import (
    ChainManager,
    ChainStatus,
    HEAD_KEY,
    HEARTBEAT_INTERVAL_S,
    ALL_LAYERS_KEY,
    EXPIRATION_S,
    DIGITS_SHOW,
)

logger = logging.getLogger(__name__)

MAX_MSG_SIZE = 100 * 1024 * 1024  # 100 MB


class Client:
    def __init__(
        self,
        model_path: Path,
        host_maddrs: List[str] = ["/ip4/0.0.0.0/tcp/4001"],
        initial_peers: List[str] = None,
        grpc_addr: str = f"{get_ip_address()}:5001",
        time_it: bool = False,
    ) -> None:
        self.grpc_addr = grpc_addr

        self.dht = DHTManager(host_maddrs=host_maddrs, initial_peers=initial_peers)
        self.dht.start()
        self.chain = ChainManager(self.dht, is_client=True)

        # Load Config
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        self.config = config

        self.llm = LLM.load(model_path, load_initial_layer=True, load_output_layer=False, time_it=time_it)
        self.model = self.llm.model
        self.chat_history = []

        self._repair_lock = threading.Lock()

        self.head_server_stub = None
        self.head_server_stub_addr = None
        self._connect_to_head()

        self.inference_response = None
        self.inference_response_event = threading.Event()
        self.final_activations = None

        self.initial_inference_delay = 0.0
        self.serialization_delay = 0.0
        self.head_communication_latency = 0.0
        self.deserialization_delay = 0.0
        self.final_inference_delay = 0.0
        self.sample_delay = 0.0
        self.decode_delay = 0.0
        self.yield_delay = 0.0

        self.total_rate = 0.0
        self.last_inference_stats = {"latency": 0.0, "throughput": 0.0}

        # Start GRPC server in a seperate thread
        grpc_thread = threading.Thread(target=self._run_grpc_server, args=(grpc_addr,), daemon=True)
        grpc_thread.start()

    def _run_grpc_server(self, grpc_addr):
        # Start GRPC server
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=1),
            options=[
                ("grpc.max_send_message_length", MAX_MSG_SIZE),
                ("grpc.max_receive_message_length", MAX_MSG_SIZE),
            ],
        )
        nodeservice_pb2_grpc.add_ClientServiceServicer_to_server(ClientServicer(self), server)
        server.add_insecure_port(grpc_addr)
        server.start()
        logger.info(f"Client is ready to accept grpc connections on {grpc_addr}.")

        head_monitor_thread = threading.Thread(target=self._head_health_monitor_task, daemon=True)
        head_monitor_thread.start()

        server.wait_for_termination()

    def _head_health_monitor_task(self):
        while True:
            time.sleep(HEARTBEAT_INTERVAL_S)
            if self.chain.get_chain_status() in (ChainStatus.READY, ChainStatus.UNREADY):
                try:
                    self._connect_to_head()
                    if self.head_server_stub is not None:
                        self.head_server_stub.Check(nodeservice_pb2.Empty(), timeout=2)
                        logger.info(f"HEAD is ALIVE.")
                except grpc.RpcError as e:
                    if (
                        e.code() == grpc.StatusCode.UNAVAILABLE
                        or e.code() == grpc.StatusCode.DEADLINE_EXCEEDED
                    ):
                        logger.warning(f"HEAD failure detected during HEALTH CHECK.")
                        self.replace_head_server()
                    else:
                        logger.warning(
                            f"A gRPC error occurred during health check: {e.code().name}"
                        )
                        self.head_server_stub = None

    def _connect_to_head(self):
        head_info = self.chain.get_head_server_info()
        if not head_info:
            self.head_server_stub = None
            return

        head_server_addr = head_info.get("address")

        # The conneection to the HEAD is fine
        if self.head_server_stub is not None and self.head_server_stub_addr == head_server_addr:
            return

        if not head_server_addr:
            logger.warning("Head Server address not found.")
            self.head_server_stub = None
            return

        logger.info(f"Client successfully found head server. Address: {head_info['address']}")

        try:
            channel = create_grpc_channel(head_server_addr)
            grpc.channel_ready_future(channel).result(timeout=10)
            self.head_server_stub = nodeservice_pb2_grpc.NodeServiceStub(channel)
            self.head_server_stub_addr = head_server_addr
            logger.info(f"Connection to head: {head_server_addr} established.")
        except grpc.FutureTimeoutError:
            logger.error(f"Connection to {head_server_addr} timed out.")
            self.head_server_stub = None
        except grpc.RpcError as e:
            logger.error(
                f"A gRPC error occurred while connecting to {head_server_addr}: {e.code().name}"
            )
            self.head_server_stub = None

    def print_chain_status(self):
        print("Chain Status:")
        pprint(self.chain.get_chain_info())

    def replace_head_server(self):
        with self._repair_lock:
            # Re-validate the failure. Another thread could have already repaired the chain
            if self.head_server_stub is not None:
                try:
                    self.head_server_stub.Check(nodeservice_pb2.Empty(), timeout=2)
                    logger.info(f"HEAD is ALIVE. Aborting unnecessary repair...")
                    return
                except grpc.RpcError as e:
                    logger.warning(f"HEAD confirmed DEAD. Proceeding with the repair...")

            # ---- HEAD Replacement ----

            dead_head_info = self.chain.get_head_server_info()

            if not dead_head_info:
                logger.error("Could not retrieve HEAD data from DHT! Chain is broken.")
                return

            head_succ_info = None
            head_succ_data = dead_head_info.get("successor")
            if head_succ_data:
                head_succ_info = self.chain.get_server_info(head_succ_data.get("id"))

            self.head_server_stub = None
            self.chain.dht.store(ALL_LAYERS_KEY, False, EXPIRATION_S)
            self.chain.update_chain_status(ChainStatus.REPAIRING)

            orphaned_layers = dead_head_info.get("layers")
            head_was_tail = self.chain.node_is_tail(dead_head_info.get("id"))
            logger.info(f"Attempting to replace HEAD. Orphaned layers: {orphaned_layers}...")

            # If the dead head had a successor, attempt to use it for the repair
            if head_succ_info:
                logger.info("Attempting to use the fail HEAD's successor for the repair...")

                head_succ_avail_mem = head_succ_info.get("available_memory")
                head_succ_avail_vram = head_succ_info.get("available_vram")

                if can_load(
                    config=self.config,
                    layers=orphaned_layers,
                    output_layer=dead_head_info.get("output_layer_loaded"),
                    avail_mem=head_succ_avail_mem,
                    avail_vram=head_succ_avail_vram,
                ):
                    new_layers = (orphaned_layers[0], head_succ_info.get("layers")[1])
                    self.chain.repair(
                        new_layers,
                        replacement_info=head_succ_info,
                        replacement_load_output_layer=head_succ_info.get("output_layer_loaded"),
                        replacee_info=dead_head_info,
                        replacee_was_head=True,
                    )

                    try:
                        self._connect_to_head()
                        self.head_server_stub.LoadLayers(nodeservice_pb2.Empty())

                        if self.chain.get_all_layers_loaded():
                            self.chain.update_chain_status(ChainStatus.READY)
                        else:
                            self.chain.update_chain_status(ChainStatus.UNREADY)
                        return
                    except grpc.RpcError as e:
                        logger.error(
                            f"A gRPC error occurred while connecting to {backup_info.get('address')}: {e.code().name}"
                        )

                    logger.info(
                        f"The dead HEAD's successor {head_succ_data.get('id')[:DIGITS_SHOW]} cannot load the orphaned layers."
                    )

            # If the head's successor couldn't load the orphaned layers or the head didn't have a successor
            # Find a backup node replacement
            logger.info("Searching for a backup node...")
            backup_nodes = self.chain.get_backup_nodes()
            if backup_nodes:
                # Find backup node with enough memory
                for backup_id in backup_nodes:
                    backup_info = self.chain.get_server_info(backup_id)
                    if backup_info:
                        backup_avail_mem = backup_info.get("available_memory")
                        backup_avail_vram = backup_info.get("available_vram")
                        logger.info(
                            f"Backup Node {backup_info.get('id')[:DIGITS_SHOW]} found with Available Memory: {backup_avail_mem} MB and Available VRAM: {backup_avail_vram} MB"
                        )

                        if can_load(
                            config=self.config,
                            avail_mem=backup_avail_mem,
                            avail_vram=backup_avail_vram,
                            layers=orphaned_layers
                        ):
                            self.chain.repair(
                                orphaned_layers,
                                replacement_info=backup_info,
                                replacement_load_output_layer=dead_head_info.get("output_layer_loaded"),
                                replacee_info=dead_head_info,
                                replacee_was_head=True,
                                replacee_was_tail=head_was_tail,
                            )

                            try:
                                self._connect_to_head()
                                self.head_server_stub.LoadLayers(nodeservice_pb2.Empty())

                                if self.chain.get_all_layers_loaded():
                                    self.chain.update_chain_status(ChainStatus.READY)
                                else:
                                    self.chain.update_chain_status(ChainStatus.UNREADY)
                                return
                            except grpc.RpcError as e:
                                logger.error(
                                    f"A gRPC error occurred while connecting to {backup_info.get('address')}: {e.code().name}"
                                )
                        else:
                            logger.info(f"Backup Node {backup_id} cannot load the orphaned layers.")

            # Neither a backup nor the head's successor can replace the dead HEAD
            logger.error(
                f"Neither a backup nor the dead head's successor can replace the dead HEAD."
            )

            # Make the dead head's successor the new HEAD and reallocate the layers among the remaining nodes
            # The chain will have to wait for a new node to come in
            if head_succ_info:
                logger.error(
                    f"Making the dead HEAD's successor the new head and reallocating layers..."
                )

                self.chain.update_chain_head(head_succ_info.get("id"))
                self._connect_to_head()

                # Reallocate among the remaining nodes
                self.trigger_reallocation()
            else:
                # Its over make the HEAD None and a backup will take its place
                # or a new node will join
                self.chain.dht.store(HEAD_KEY, None, EXPIRATION_S)
                self.chain.update_chain_status(ChainStatus.UNREADY)

    def trigger_reallocation(self):
        """
        Triggers the layer reallocation process starting from the HEAD.
        """
        # Gather the total rate from all the server nodes in case a node has failed in the time between the previous generation and the reallocation
        self.total_rate = self.chain.gather_total_rate()

        if self.total_rate > 0:
            logger.info(
                f"Triggering reallocation with Total Rate: {self.total_rate:.2f} layers/sec..."
            )

            request = nodeservice_pb2.ReallocateRequest(
                total_rate=self.total_rate, start_layer_index=0
            )

            try:
                self.chain.update_chain_status(ChainStatus.REALLOCATING)

                self.head_server_stub.Reallocate(request)

                self.chain.update_all_layers_loaded()
                if self.chain.get_all_layers_loaded():
                    self.chain.update_chain_status(ChainStatus.READY)
                else:
                    self.chain.update_chain_status(ChainStatus.UNREADY)

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
        chain_status = self.chain.get_chain_status()
        if not all_layers_loaded or chain_status != ChainStatus.READY:
            warning = '<span style="color:red">Not all model layers have been loaded or the chain is not READY. Cannot initiate the generation task.</span>'
            logger.warning(warning)
            return warning

        # Connect to head again. Successor stub could be stale after an Opportunistic Takeover
        self._connect_to_head()

        prompt = self.llm.apply_chat_template(prompt)
        self.chat_history.append(prompt)

        full_prompt = "".join(self.chat_history)
        input_ids = self.llm.preprocessor.encode(full_prompt)

        prompt_length = input_ids.size(1)
        max_returned_tokens = prompt_length + max_new_tokens

        if max_returned_tokens > self.model.max_seq_length:
            raise ValueError(
                f"The combined prompt and max_new_tokens length ({max_returned_tokens}) exceeds "
                f"the model's maximum sequence length of {self.model.max_seq_length}."
            )

        if stream:
            return self._generate_stream(
                prompt_length,
                input_ids,
                max_new_tokens,
                max_returned_tokens,
                temperature,
                top_p,
                time_it,
            )

        # If not streaming the output
        decoded_text = self._generate_fn(
            prompt_length,
            input_ids,
            max_new_tokens,
            max_returned_tokens,
            temperature,
            top_p,
            time_it,
        )
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

        start_time = time.perf_counter()
        for _ in range(max_new_tokens):
            # logger.info(f"Generating token {i + 1}/{max_new_tokens}")  # Debugging
            x = self.model.forward_client_initial(input_tensor, input_pos=input_pos)

            # Call the remote server chain
            request = tensor_to_request(
                x,
                max_returned_tokens=max_returned_tokens,
                seq_length=seq_length,
                input_pos=input_pos.item() if input_pos is not None else None,
            )
            request.response_address = self.grpc_addr

            ack_response = self.head_server_stub.RunLayers(request)

            if ack_response.HasField("error_message"):
                logger.error(f"Server-side failure: {ack_response.error_message}")
                logger.error("Aborting generation task. Please try again.")
                return

            is_set = self.inference_response_event.wait(timeout=15)
            if not is_set:
                logger.error("Timeout waiting for response from Tail server.")

            response = self.inference_response

            # Capture TOTAL RATE
            self.total_rate = self.chain.gather_total_rate()

            x = message_to_tensor(response)

            # Run clients final layers
            logits = self.model.forward_client_final(x)

            # Sample the next token
            next_token = self.llm.sample_logits(logits, temperature, top_p)

            # Stop if the end-of-sequence token is generated
            if next_token.item() == self.llm.preprocessor.tokenizer.eos_token_id:
                self.chat_history.append(self.llm.preprocessor.tokenizer.eos_token)
                break

            generated_ids.append(next_token)
            input_tensor = next_token
            current_pos = prompt_length + (len(generated_ids))
            input_pos = torch.tensor([current_pos], device=self.llm.preprocessor.device)
            seq_length = 1

        elapsed_time = time.perf_counter() - start_time
        throughput = len(generated_ids) / elapsed_time if elapsed_time > 0 else 0

        all_generated_ids = torch.cat(generated_ids, dim=1)
        all_generated_tokens = self.llm.preprocessor.decode(all_generated_ids)

        self.last_inference_stats = {"latency": elapsed_time, "throughput": throughput}
        self.chain.update_chain_status(ChainStatus.READY)

        return all_generated_tokens

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
            start_token_gen = time.perf_counter()
            x = self.model.forward_client_initial(input_tensor, input_pos=input_pos)
            self.initial_inference_delay = time.perf_counter() - start_token_gen

            # Call the remote server chain
            start = time.perf_counter()
            request = tensor_to_request(
                x,
                max_returned_tokens=max_returned_tokens,
                seq_length=seq_length,
                input_pos=input_pos.item() if input_pos is not None else None,
            )
            request.response_address = self.grpc_addr

            self.serialization_delay = time.perf_counter() - start

            start = time.perf_counter()
            try:
                ack_response = self.head_server_stub.RunLayers(request)
            except grpc.RpcError as e:
                if e.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
                    logger.warning(f"HEAD failure detected duting INFERENCE")
                    threading.Thread(target=self.replace_head_server, daemon=True).start()
                else:
                    logger.warning(f"A gRPC error occurred during inference: {e.code().name}")
                    self.head_server_stub = None

                yield f"<br><span style='color:red'>The HEAD server has failed. The chain is being repaired..."
                return

            end = time.perf_counter()

            total_rpc_time = end - start
            # logger.info(f"Total HEAD RPC Delay: {total_rpc_time:.6f}")
            if ack_response.processing_time > 0:
                total_rpc_time = end - start
                current_network_latency = total_rpc_time - ack_response.processing_time

                if self.head_communication_latency > 0.0:
                    self.head_communication_latency = (0.7 * self.head_communication_latency) + (
                        0.3 * current_network_latency
                    )
                else:
                    self.head_communication_latency = current_network_latency

                # logger.info(f"Head Communication Latency: {self.head_communication_latency:.6f}s")

            if ack_response.HasField("error_message"):
                logger.error(f"Server-side failure: {ack_response.error_message}")
                logger.error("Aborting generation task. Please try again.")

                if self.chain.get_all_layers_loaded():
                    self.chain.update_chain_status(ChainStatus.READY)

                yield f'<br><span style="color:red">Server-side failure: {ack_response.error_message} Aborting generation task. Please try again.</span>'
                return

            # Wait for the Tail to set the response_event
            is_set = self.inference_response_event.wait(timeout=15)

            if not is_set:
                logger.error("Timeout waiting for response from Tail server.")
                if self.chain.get_all_layers_loaded():
                    self.chain.update_chain_status(ChainStatus.READY)

                yield "Error: Timeout"
                return

            response = self.inference_response

            # Capture the TOTAL PROCESSING RATE
            self.total_rate = self.chain.gather_total_rate()

            start = time.perf_counter()
            next_token = response_to_tensor(response)
            self.deserialization_delay = time.perf_counter() - start

            # Stop if the end-of-sequence token is generated
            if next_token.item() == self.llm.preprocessor.tokenizer.eos_token_id:
                self.chat_history.append(self.llm.preprocessor.tokenizer.eos_token)
                break

            # Decode and yield the new token
            start = time.perf_counter()
            decoded_token = self.llm.preprocessor.decode(next_token)
            self.decode_delay = time.perf_counter() - start

            tokens_generated += 1

            self.chat_history.append(decoded_token)
            start = time.perf_counter()
            yield decoded_token
            self.yield_delay = time.perf_counter() - start

            input_tensor = next_token
            current_pos = prompt_length + (i + 1)
            input_pos = torch.tensor([current_pos], device=self.llm.preprocessor.device)
            seq_length = 1

        elapsed_time = time.perf_counter() - start_time
        throughput = tokens_generated / elapsed_time if elapsed_time > 0 else 0

        self.last_inference_stats = {"latency": elapsed_time, "throughput": throughput}
        self.chain.update_chain_status(ChainStatus.READY)
