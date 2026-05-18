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
from core.utils import can_load, calculate_model_size_mb
from core.constants import *
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
    NUM_CLIENTS_KEY,
    EXPIRATION_S,
    DIGITS_SHOW,
)

logger = logging.getLogger(__name__)


class Client:
    def __init__(
        self,
        grpc_addr: str,
        model_path: Path,
        host_maddrs: List[str] = ["/ip4/0.0.0.0/tcp/4001"],
        initial_peers: List[str] = None,
        time_it: bool = False,
    ) -> None:
        # Set the device
        self.device = "cpu"
        if torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
                free_mb = free_bytes / (1024 * 1024)
                if free_mb >= RESERVED_MEM_MB:
                    self.device = "cuda"
            except RuntimeError as e:
                logger.warning(f"Failed to allocate CUDA context, defaulting to CPU. Error: {e}")

        self.grpc_addr = grpc_addr

        self.dht = DHTManager(host_maddrs=host_maddrs, initial_peers=initial_peers)
        self.dht.start()
        self.chain = ChainManager(self.dht, is_client=True)
        self.chain.increment_num_clients()
        self.chain.add_node_to_clients(self.chain.node_id)

        # Load Config
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        self.config = config

        self.llm = LLM.load(model_path, device=self.device, load_initial_layer=True, load_output_layer=False, time_it=time_it)
        self.model = self.llm.model
        self.chat_history = []

        self._repair_lock = threading.Lock()

        self.head_server_stub = None
        self.head_server_stub_addr = None
        self._connect_to_head()

        self.inference_response: nodeservice_pb2.InferenceResponse = None
        self.inference_response_event = threading.Event()

        self.initial_inference_delay = 0.0
        self.serialization_delay = 0.0
        self.head_communication_latency = 0.0
        self.deserialization_delay = 0.0
        self.decode_delay = 0.0
        self.yield_delay = 0.0

        self.total_rate = 0.0
        self.last_inference_stats = {"latency": 0.0, "throughput": 0.0}

        # Start GRPC server in a seperate thread
        grpc_thread = threading.Thread(target=self._run_grpc_server, args=(grpc_addr,), daemon=True)
        grpc_thread.start()
        self.grpc_server = None
        
        self._stop_health_monitor_event = threading.Event()
        self._stop_dht_heartbeat_task = threading.Event()

    def _run_grpc_server(self, grpc_addr):
        # Start GRPC server
        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=1),
            options=[
                ("grpc.max_send_message_length", GRPC_MAX_MSG_SIZE),
                ("grpc.max_receive_message_length", GRPC_MAX_MSG_SIZE),
            ],
        )
        nodeservice_pb2_grpc.add_ClientServiceServicer_to_server(ClientServicer(self), self.grpc_server)
        self.grpc_server.add_insecure_port(grpc_addr)
        self.grpc_server.start()
        logger.info(f"Client is ready to accept grpc connections on {grpc_addr}.")

        threading.Thread(target=self._head_health_monitor_task, daemon=True).start()
        threading.Thread(target=self._dht_heartbeat_task, daemon=True).start()

        self.grpc_server.wait_for_termination()
    
    def shutdown(self):
        # Stop the head health monitor and dht heartbeat
        self._stop_health_monitor_event.set()
        self._stop_dht_heartbeat_task.set()

        # Decrement the num_clients DHT key
        self.chain.decrement_num_clients()

        # Remove form client_nodes list
        self.chain.remove_node_from_clients(self.chain.node_id)

        # Remove this client's kv cache from the server nodes
        num_clients = self.chain.dht.get(NUM_CLIENTS_KEY)
        max_seq_length = GLOBAL_MAX_SEQ_LEN // num_clients if num_clients else GLOBAL_MAX_SEQ_LEN

        request = nodeservice_pb2.RemoveClientKVCacheRequest(
            client_id=self.grpc_addr,
            new_max_seq_length=max_seq_length
        )

        self.head_server_stub.RemoveClientKVCache(request)
        
        # Shut down the P2P/DHT connections and GRPC server
        if self.grpc_server:
            logger.info("Shutting down GRPC server...")
            self.grpc_server.stop(grace=None)
        
        if hasattr(self, 'dht') and self.dht:
            self.dht.shutdown()

    def _dht_heartbeat_task(self):
        """Background task to keep client DHT keys alive."""
        while not self._stop_health_monitor_event.is_set():
            is_stopped = self._stop_health_monitor_event.wait(timeout=HEARTBEAT_INTERVAL_S)
            if is_stopped:
                break

            self.chain.republish_client_keys()

    def _head_health_monitor_task(self):
        while not self._stop_health_monitor_event.is_set():
            # time.sleep(HEARTBEAT_INTERVAL_S)
            is_stopped = self._stop_health_monitor_event.wait(timeout=HEARTBEAT_INTERVAL_S)
            if is_stopped:
                break

            if self.chain.get_chain_status() in (ChainStatus.READY, ChainStatus.UNREADY, ChainStatus.REPAIRING):
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
            
            self.head_server_stub = None
            
            # Only one client has to do the repair
            client_nodes = self.chain.get_client_nodes()

            if client_nodes[0] != self.chain.node_id:
                logger.info(f"Waiting for client {client_nodes[0][:DIGITS_SHOW]} to finish the HEAD replacement...")

                while self.chain.get_chain_status() == ChainStatus.REPAIRING:
                    time.sleep(HEARTBEAT_INTERVAL_S)
                
                self._connect_to_head()
                return

            self.chain.dht.store(ALL_LAYERS_KEY, False, EXPIRATION_S)
            # self.chain.become_chain_leader()
            self.chain.update_chain_status(ChainStatus.REPAIRING)

            head_succ_info = None
            head_succ_data = dead_head_info.get("successor")
            if head_succ_data:
                head_succ_info = self.chain.get_server_info(head_succ_data.get("id"))


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
                    output_layer_memory_tle=head_succ_info.get("output_layer_memory_tle")
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
                            output_layer_memory_tle=backup_info.get("output_layer_memory_tle"),
                            layers=orphaned_layers,
                            output_layer=dead_head_info.get("output_layer_loaded"),
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
                # Its over make the HEAD None 
                self.chain.dht.store(HEAD_KEY, None, EXPIRATION_S)
                self.chain.update_all_layers_loaded()
            
            # Use backup nodes to rebuild the chain 
            logger.info(f"Attempting to rebuild the chain with backup nodes...")
            backup_nodes = self.chain.get_backup_nodes()
            if backup_nodes:
                self.chain.update_chain_status(ChainStatus.REPAIRING)
                # Find backup node with enough memory
                for backup_id in backup_nodes:
                    backup_info = self.chain.get_server_info(backup_id)
                    
                    backup_addr = backup_info.get("address")
                    if not backup_addr:
                        continue

                    try:
                        channel = create_grpc_channel(backup_addr)
                        backup_stub = nodeservice_pb2_grpc.NodeServiceStub(channel)
                        backup_stub.JoinChain(nodeservice_pb2.Empty())
                    except grpc.RpcError as e:
                        logger.error(
                            f"A gRPC error occurred while connecting to {backup_addr}: {e.code().name}"
                        )
                
                self._connect_to_head()
            
            if self.chain.get_all_layers_loaded():
                self.chain.update_chain_status(ChainStatus.READY)
            else:
                self.chain.update_chain_status(ChainStatus.UNREADY)

            

    def trigger_reallocation(self):
        """
        Triggers the layer reallocation process starting from the HEAD.
        """
        # Gather the total rate from all the server nodes in case a node has failed in the time between the previous generation and the reallocation
        # self.chain.become_chain_leader()
        # self.chain.update_chain_status(ChainStatus.REALLOCATING)

        requires_restart = True
        while requires_restart:
            self.chain.update_chain_status(ChainStatus.REALLOCATING)

            self.total_rate = self.chain.gather_total_rate()
            
            # Check if the model can be loaded on the available active server nodes
            model_mem_size, layer_mem_size, output_layer_mem_size = calculate_model_size_mb(self.config)
            total_chain_memory = self.chain.gather_total_memory(layer_mem_size, output_layer_mem_size)

            load_max = False
            if total_chain_memory < model_mem_size:
                load_max = True
                logger.info("The nodes in the active chain can't load the full model. Loading the maximum number of layers...")

            if self.total_rate <= 0:
                logger.error("Total Rate is 0 or less, cannot reallocate.")
                break

            logger.info(
                f"Triggering reallocation with Total Rate: {self.total_rate} layers/sec..."
            )

            request = nodeservice_pb2.ReallocateRequest(
                total_rate=self.total_rate, 
                start_layer_index=0,
                load_max=load_max
            )

            try:
                response = self.head_server_stub.Reallocate(request)

                if response.requires_restart:
                    logger.info(f"Reallocation bottlenecked at {response.bottleneck_node}. Restarting...")
                    continue

                if response.success:
                    self.chain.update_all_layers_loaded()
                    
                    logger.info("Reallocation completed successfully.")
                    break
                
                else:
                    logger.error("Reallocation failed without a restart request.")
                    break

            except grpc.RpcError as e:
                logger.error(f"Failed to trigger reallocation: {e}")
                break
        
        if self.chain.get_all_layers_loaded():
            self.chain.update_chain_status(ChainStatus.READY)
        else:
            self.chain.update_chain_status(ChainStatus.UNREADY)

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
        if not all_layers_loaded or chain_status not in (ChainStatus.READY, ChainStatus.RUNNING):
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

        num_clients = self.chain.dht.get(NUM_CLIENTS_KEY)
        if max_returned_tokens > GLOBAL_MAX_SEQ_LEN // num_clients:
            return (
                f"<span style='color:red'>The combined prompt and max new tokens length ({max_returned_tokens} tokens) exceeds this client's max sequence length of {GLOBAL_MAX_SEQ_LEN//num_clients} tokens.<br>Please clear the chat history or reduce the number of new tokens to generate."
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
            x = self.model.forward_client_initial(input_tensor, input_pos=input_pos)

            x = x.cpu()

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
                if self.chain.get_all_layers_loaded():
                    self.chain.update_chain_status(ChainStatus.READY)
                return

            response = self.inference_response

            # Capture TOTAL RATE
            self.total_rate = self.chain.gather_total_rate()

            next_token = response_to_tensor(response)

            if next_token.device != self.llm.device:
                next_token = next_token.to(self.llm.device)

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
        # self.chain.become_chain_leader()
        self.chain.update_chain_status(ChainStatus.RUNNING)

        input_tensor = input_ids
        input_pos = None
        seq_length = prompt_length

        start_time = time.perf_counter()
        tokens_generated = 0
        for i in range(max_new_tokens):
            # self.chain.become_chain_leader()

            start_token_gen = time.perf_counter()
            x = self.model.forward_client_initial(input_tensor, input_pos=input_pos)
            self.initial_inference_delay = time.perf_counter() - start_token_gen

            # Move output tensor back to the cpu for serialization
            x = x.cpu()

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

            try:
                start = time.perf_counter()
                self.head_server_stub.RunLayers(request)

                self.head_communication_latency = time.perf_counter() - start
            except grpc.RpcError as e:
                if e.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
                    logger.warning(f"HEAD failure detected duting INFERENCE")
                    threading.Thread(target=self.replace_head_server, daemon=True).start()
                else:
                    logger.warning(f"A gRPC error occurred during inference: {e.code().name}")
                    self.head_server_stub = None

                yield f"<br><span style='color:red'>The HEAD server has failed. The chain is being repaired..."
                self.chain.update_chain_status(ChainStatus.UNREADY)
                return
            

            # Wait for the Tail to set the inference_response event
            is_set = self.inference_response_event.wait(timeout=15)

            if not is_set:
                logger.error("Timeout waiting for response from Tail server.")
                # if self.chain.get_all_layers_loaded():
                #     self.chain.update_chain_status(ChainStatus.READY)

                yield f'<br><span style="color:red">Error: Timeout'
                self.chain.update_chain_status(ChainStatus.UNREADY)
                return

            self.inference_response_event.clear()
            response = self.inference_response

            # Error handling
            if response.HasField("error_message"):
                logger.error(f"Server-side failure: {response.error_message}")
                logger.error("Aborting generation task. Please try again.")

                # if self.chain.get_all_layers_loaded():
                #     self.chain.update_chain_status(ChainStatus.READY)

                yield f'<br><span style="color:red">Server-side failure: {response.error_message} Aborting generation task. Please try again.</span>'
                self.chain.update_chain_status(ChainStatus.UNREADY)
                return

            # Capture the TOTAL PROCESSING RATE
            self.total_rate = self.chain.gather_total_rate()

            start = time.perf_counter()
            next_token = response_to_tensor(response)
            self.deserialization_delay = time.perf_counter() - start

            if next_token.device != self.llm.device:
                next_token = next_token.to(self.llm.device)

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

        self.last_inference_stats = {"num_tokens_generated": tokens_generated, "latency": elapsed_time, "throughput": throughput}
        # self.chain.update_chain_status(ChainStatus.READY)
        if self.chain.get_all_layers_loaded():
            self.chain.update_chain_status(ChainStatus.READY)
        else:
            self.chain.update_chain_status(ChainStatus.UNREADY)
