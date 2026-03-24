"""Main server application logic"""

import json
import logging
import os
import signal
import threading
import socket
from dotenv import load_dotenv
import gc
import time
from concurrent import futures
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import grpc
import psutil
import torch

from servicer import NodeServicer
from core.llm_loader import LLM
from core.remote.utils import get_bootstrap_peer_address, get_ip_address, discover_bootstrap_node_address
from core.remote import nodeservice_pb2, nodeservice_pb2_grpc
from core.remote.serialization import *
from core.p2p.dht_manager import DHTManager
from core.p2p.chain_manager import ChainManager, HEARTBEAT_INTERVAL_S, ALL_LAYERS_KEY, EXPIRATION_S

logger = logging.getLogger(__name__)

RESERVED_MEM_MB = 500  # Memory reserved for system overhead


class Server:
    def __init__(
        self,
        model_path: Path,
        num_layers: int = None,
        added_delay: float = None, # Debugging
        time_it: bool = False,
        host_maddrs: List[str] = ["/ip4/0.0.0.0/tcp/4001"],
        initial_peers: List[str] = None,
        grpc_addr: str = "head-server:5001",
    ) -> None:
        self.added_delay = added_delay

        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self._repair_lock = threading.Lock()

        # Load Config
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        self.config = config

        # Memory
        self.process = psutil.Process(os.getpid())
        self.memory_limit_mb = self._get_container_memory_limit_mb()
        self.memory_usage_mb = 0.0
        self.available_memory_mb = 0.0

        # VRAM
        self.vram_usage_mb = 0.0
        self.vram_limit_mb = 0.0
        self.available_vram_mb = 0.0

        # Create a DHT Node and Chain object
        self.dht = DHTManager(host_maddrs=host_maddrs, initial_peers=initial_peers)
        self.dht.start()
        self.chain = ChainManager(self.dht)

        # Determine Load Capacity
        self._update_memory_usage(update_on_dht=False)
        if not num_layers:
            num_layers = self._mem_to_num_layers(bytes_per_param=4)
            logger.info(f"Node with {self.available_memory_mb} MB available can load {num_layers} layers.")
        elif not self._can_load(num_layers=num_layers):
            logger.warning(f"Requested {num_layers} layers but memory is insufficient. Adjusting...")
            num_layers = self._mem_to_num_layers(bytes_per_param=4)

        # Join the Inference Chain
        server_info = {"address": grpc_addr}

        self.chain.join_chain(
            server_info,
            max_num_layers=num_layers,
            num_total_layers=config["num_hidden_layers"],
        )

        # Initialize State
        self.llm = None
        self.model = None
        self.successor_stub = None
        self.num_local_layers = 0

        self.inference_delay = 0.0
        self.layers_per_second = 0.0
        self.grpc_overhead = 0.0
        # self.succ_network_latency = 0.0

        # Client Stub
        self.client_stub = None

        # If all layers are loaded then this node acts as a backup and doesn't load any layers
        if self.chain.is_backup():
            self._update_memory_usage()
            return

    def _load_llm(
        self,
        time_it: bool = False,
    ):
        """Loads the layers assigned by the ChainManager."""
        if self.chain.is_backup():
            self._profile_backup_node()
            return

        layers_to_loaded = self.chain.get_layers()
        logger.info(f"Loading layers: {layers_to_loaded}...")

        self.llm = LLM.load(
            self.model_path,
            is_client=False,
            layers_to_load=layers_to_loaded,
            time_it=time_it,
        )
        self.model = self.llm.model
        self.num_local_layers = self.llm.model.num_layers

        # Update DHT and Chain Status
        self.chain.update_layers_loaded(True)
        self.chain.update_all_layer_loaded()
        self.chain.update_device(self.llm.device)
        self._update_memory_usage()

    def _mem_to_num_layers(
            self,
            bytes_per_param: int, 
            avail_mem: float = None, 
            avail_vram: float = None,
        ) -> int:
        """
        Calculates how many layers fit in the available memory/VRAM
        """
        self._update_memory_usage(update_on_dht=False)

        total_layer_params = self.config["total_transformer_layer_params"]
        layer_memory_size_mb = (total_layer_params * bytes_per_param) / (1024 * 1024)

        # If avail_mem or avail_vram is given, max_num_layers is requested for a different node that this one
        if avail_mem or avail_vram:
            if avail_vram:
                available = avail_vram
            else:
                available = avail_mem

        elif self.device == "cuda":
            available = self.available_vram_mb - RESERVED_MEM_MB
        else:
            available = self.available_memory_mb - RESERVED_MEM_MB

        if available <= 0:
            return 0

        max_num_layers = int(available // layer_memory_size_mb)
        return min(max_num_layers, self.config["num_hidden_layers"])

    def _can_load(
        self,
        num_layers: Optional[int] = None,
        layers: Optional[Tuple[int, int]] = None,
        avail_mem: float = None,
        avail_vram: float = None,
    ) -> bool:
        """Checks if this node can load a certain number of layers or a range of layers"""
        num_layers = num_layers if num_layers else (layers[1] - layers[0] + 1)
        max_num_layers = self._mem_to_num_layers(bytes_per_param=4, avail_mem=avail_mem, avail_vram=avail_vram)
        return num_layers <= max_num_layers

    def _get_container_memory_limit_mb(self) -> Optional[float]:
        """Reads the container's memory limit from cgroup files."""
        cgroup_v2_path = "/sys/fs/cgroup/memory.max"

        try:
            if os.path.exists(cgroup_v2_path):
                with open(cgroup_v2_path, "r") as f:
                    content = f.read().strip()
                    if content and content != "max":
                        return int(content) / (1024 * 1024)
        except (FileNotFoundError, PermissionError, ValueError):
            pass

        return 0.0

    def _update_memory_usage(self, update_on_dht: bool = True):
        # ---- Memory ----
        # Get current memory usage in MB
        memory_bytes = self.process.memory_info().rss
        self.memory_usage_mb = memory_bytes / (1024 * 1024)

        # Get the container's memory
        if self.memory_limit_mb:
            self.available_memory_mb = self.memory_limit_mb - self.memory_usage_mb
        else:
            # Fallback to host's available memory if no limit is set
            available_bytes = psutil.virtual_memory().available
            self.available_memory_mb = available_bytes / (1024 * 1024)

        # ---- VRAM ----
        if self.device == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            self.vram_limit_mb = total_bytes / (1024 * 1024)
            self.available_vram_mb = free_bytes / (1024 * 1024)
            self.vram_usage_mb = self.vram_limit_mb - self.available_vram_mb

        if update_on_dht:
            self.chain.update_memory(self.memory_usage_mb, self.memory_limit_mb, self.available_memory_mb)
            
            if self.device == "cuda":
                self.chain.update_vram(self.vram_usage_mb, self.vram_limit_mb, self.available_vram_mb)
    
    @torch.no_grad()
    def _profile_backup_node(self):
        """ Measures a backup node's processing rate by doing an inference run on a dummy input."""
        # Load the first Transformer layer
        self.llm = LLM.load(
            self.model_path,
            is_client=False,
            layers_to_load=(0,0)
        )
        self.model = self.llm.model

        self.chain.update_device(self.llm.device)
        self._update_memory_usage()

        # Inference with dummy input
        hidden_size = self.config["hidden_size"]
        seq_len = 30 # short sequence for profiling

        # Profiling parameters
        profiling_runs = 100

        dummy_input = torch.randn(1, seq_len, hidden_size, device=self.llm.device, dtype=torch.float32)
        
        start = time.perf_counter()
        for _ in range(profiling_runs):
            self.model.forward_server(dummy_input, seq_length=seq_len, input_pos=None)
            if self.added_delay:
                time.sleep(self.added_delay)
        elapsed = time.perf_counter() - start

        avg_latency = elapsed / profiling_runs
        processing_rate = self.model.num_layers / avg_latency
        logger.info(f"Backup profiling: avg latency {avg_latency:.6f}s/run, layers/sec {processing_rate:.2f}")

        self.chain.update_processing_rate(processing_rate)
        

    @torch.no_grad()
    def run_local_layers(
        self,
        input_tensor: torch.Tensor,
        max_returned_tokens: int,
        seq_length: int = None,
        input_pos: torch.Tensor = None,
        incoming_partial_rate: float = 0.0,
        response_address: str = None,
    ) -> nodeservice_pb2.InferenceResponse:
        """
        This function runs inference on the server's assigned transformer layers and send the output to the next node.
        """
        if not self.llm:
            return nodeservice_pb2.InferenceResponse(error_message="A node in the chain does not have its layers loaded.")

        # Ensure inputs are on the same device as the model
        device = self.llm.device

        if input_tensor.device != device:
            input_tensor = input_tensor.to(device)

        if input_pos is not None and input_pos.device != device:
            input_pos = input_pos.to(device)

        # KV Cache
        if not self.llm.kv_cache_initialized:
            # Na allaksw to batch_size otan kanw batched inference
            self.model.set_kv_cache(batch_size=1, max_seq_length=max_returned_tokens, device=device)
            self.llm.kv_cache_initialized = True
            self._update_memory_usage(update_on_dht=True)

        # Dynamically grow the kv cache size if necessary
        elif self.llm.prev_generated_seq_length < max_returned_tokens:
            tmp_device = self.model.mask_cache.device
            self.model.clear_kv_cache()
            self.model.set_kv_cache(batch_size=1, max_seq_length=max_returned_tokens, device=tmp_device)
            self._update_memory_usage(update_on_dht=True)

        self.llm.prev_generated_seq_length = max_returned_tokens

        # logger.info(f"Processing layers {self.llm.layers_loaded}...")  # Debugging

        # Inference
        start = time.perf_counter()

        h = self.model.forward_server(input_tensor, seq_length, input_pos)
        if self.added_delay:
            time.sleep(self.added_delay)

        self.inference_delay = time.perf_counter() - start

        # Get the server's layer processing rate for this inference run
        if self.inference_delay > 0 and self.num_local_layers > 0:
            current_rate = self.num_local_layers / self.inference_delay

            # Moving average to avoid jitter
            if self.layers_per_second == 0:
                self.layers_per_second = current_rate
            else:
                self.layers_per_second = (0.7 * self.layers_per_second) + (0.3 * current_rate)

        logger.info(f"Layers {self.llm.layers_loaded}: " f"Delay: {self.inference_delay:.4f}s " f"Layers/sec: {self.layers_per_second:.2f} ")

        # Calculate partial rate to send forward
        my_partial_rate = incoming_partial_rate + self.layers_per_second

        # Move output tensor back to the cpu for serialization
        h = h.cpu()

        # To prevent nonesense output when a node fails during inference
        if not self.chain.get_all_layers_loaded():
            return nodeservice_pb2.InferenceResponse(error_message="Inference requested without all the layers being loaded...")

        # If TAIL Node -> Send response to Client
        if self.chain.is_tail():
            if not response_address:
                logger.error("Tail node has no response_address for the client!")
                return
            
            response = tensor_to_response(h)
            response.total_rate = my_partial_rate

            # Connect to Client
            try:
                if not self.client_stub:
                    channel = grpc.insecure_channel(response_address)
                    self.client_stub = nodeservice_pb2_grpc.ClientServiceStub(channel)

                client_response = self.client_stub.ReceiveResponse(response)
                
            except grpc.RpcError as e:
                logger.error(f"Failed to send result to client at {response_address}: {e}")

            return nodeservice_pb2.InferenceResponse(processing_time=client_response.processing_time, total_rate=0.0)

        # INTERMEDIATE NODE -> FORWARD TO SUCCESSOR

        # Ensure the successor stub has been created
        if not self.successor_stub and not self.chain.is_tail():
            self._connect_to_successor()

        # Call the successor via gRPC
        request = tensor_to_request(
            h,
            max_returned_tokens=max_returned_tokens,
            seq_length=seq_length,
            input_pos=input_pos.item() if input_pos is not None else None,
        )

        request.partial_rate = my_partial_rate
        request.response_address = response_address

        try:
            response = self.successor_stub.RunLayers(request, timeout=2)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE or e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                logger.warning(f"Successor failure detected during INFERENCE.")
                dead_successor_data = self.chain.get_failed_successor_data()
                if dead_successor_data:
                    self.chain.dht.store(ALL_LAYERS_KEY, False, EXPIRATION_S)
                    threading.Thread(target=self.repair_chain, daemon=True).start()
                    return nodeservice_pb2.InferenceResponse(error_message="A node in the chain has failed. The chain is being repaired...")
                else:
                    logger.error(f"Could not retrieve successor data from DHT! Chain is broken.")
                    return nodeservice_pb2.InferenceResponse(
                        error_message="A node in the chain failed and its data could not be retrieved. Chain is broken."
                    )

        # return nodeservice_pb2.InferenceResponse(processing_time=response.processing_time)
        return response

    def reallocate_layers(self, total_system_rate: float, start_layer_index: int):
        """
        Executes the AR-MDI logic
        """
        logger.info(
            f" REALLOCATION TRIGGERED:\nTotal Rate: {total_system_rate:.2f} | My Rate: {self.layers_per_second:.2f} | Start Index: {start_layer_index}"
        )

        num_total_layers = self.config["num_hidden_layers"]

        # Calculate share
        # layers_to_load = num_total_layers * (my_rate / total_rate)
        if total_system_rate > 0:
            # Epic equation
            ideal_layer_count = num_total_layers * (self.layers_per_second / total_system_rate)
        else:
            ideal_layer_count = 0

        logger.info(f"Idead Layer Count: {ideal_layer_count}")

        current_num_layers = self.num_local_layers
        max_extra_num_layers = self._mem_to_num_layers(bytes_per_param=4)
        max_num_layers = current_num_layers + max_extra_num_layers

        logger.info(f"Max Num Layers: {max_num_layers}")

        target_layer_count = min(int(round(ideal_layer_count)), max_num_layers)
        # target_layer_count = max_num_layers if max_num_layers > ideal_layer_count else ideal_layer_count

        # Rounding logic
        if self.chain.is_tail():
            # The tail takes whatever is left
            target_layer_count = num_total_layers - start_layer_index
        elif target_layer_count < 1:
            target_layer_count = 1
        elif (start_layer_index + target_layer_count) >= num_total_layers:
            # Dont exceed the available layers
            target_layer_count = num_total_layers - start_layer_index #- 1

        # if target_layer_count > 0:
        end_layer_index = start_layer_index + target_layer_count - 1
        new_layers = (start_layer_index, end_layer_index)
        logger.info(f"Target Layer Count: {target_layer_count} | New Layers: {new_layers}")

        # Load Layers
        if new_layers != self.llm.layers_loaded:
            self._reload_llm(new_layers)
            self.chain.update_layers(new_layers)
        else:
            logger.info("Layer assignment unchanged.")

        next_start_index = end_layer_index + 1
        # else:
        #     # Set this node as a backup
        #     logger.info(f"Target Layer Count: {target_layer_count}. Setting this node as a backup")
        #     # self.successor_stub = None
        #     self.llm = None
        #     self.num_local_layers = 0
        #     self.chain.become_backup()

        #     next_start_index = start_layer_index

        # Propagate to successor
        if not self.chain.is_tail() and self.successor_stub:
            # next_start_index = end_layer_index + 1
            request = nodeservice_pb2.ReallocateRequest(total_rate=total_system_rate, start_layer_index=next_start_index)
            try:
                self.successor_stub.Reallocate(request)
            except grpc.RpcError as e:
                logger.error(f"Failed to propagate Reallocation to successor: {e}")

    def repair_chain(self):
        """
        Method that repairs the inference chain after detecting this node's successor is dead.
        To repair the chain:
        1. If this node can hold the orphaned layers of the failed node, it loads them.
        2. Else, if a backup node is available it prompts it to take the place of the failed node.
        3. Otherwise, the chain can't be repaired and the repairing node becomes the new tail.
        """
        with self._repair_lock:
            # Re-validate the failure. Another thread could have already repaired the chain
            if self.successor_stub is not None:
                try:
                    self.successor_stub.Check(nodeservice_pb2.Empty(), timeout=2)
                    logger.info("Successor is ALIVE. Aborting unnecessary repair...")
                    return
                except grpc.RpcError as e:
                    logger.warning(f"Successor confirmed DEAD. Proceeding with repair...")

            dead_successor_data = self.chain.get_failed_successor_data()

            if not dead_successor_data:
                logger.error("Could not retrieve successor data from DHT! Chain is broken.")
                return

            self.successor_stub = None
            self.chain.dht.store(ALL_LAYERS_KEY, False, EXPIRATION_S)  # 'all_layers_loaded' -> False

            # Repair
            orphaned_layers = dead_successor_data.get("layers")
            successor_2_data = dead_successor_data.get("successor")
            succ_was_tail = dead_successor_data.get("was_tail")
            logger.info(
                f"Attempting to repair chain. Orphaned layers: {orphaned_layers}, Successor^2 Data: {successor_2_data}, Was TAIL: {succ_was_tail}"
            )

            # Check if this node has enough memory to load the orphaned layers
            if self._can_load(layers=orphaned_layers):
                layers_to_load = (self.llm.layers_loaded[0], orphaned_layers[1])
                logger.info(f"Taking over layers {orphaned_layers}. New range: {layers_to_load}")

                self._reload_llm(layers_to_load)
                self.chain.repair(layers_to_load, successor_2_data, succ_was_tail)
                self._connect_to_successor()

            # Else, try to find a backup node to take over
            else:
                logger.info("Cannot load layers. Searching for a backup node...")
                backup_nodes = self.chain.get_backup_nodes()
                if backup_nodes:
                    # Find backup node with enough memory
                    for backup_id in backup_nodes:
                        backup_info = self.chain.get_server_info(backup_id)
                        if backup_info:
                            avail_mem = backup_info.get("available_memory")
                            avail_vram = backup_info.get("available_vram")
                            
                            if self._can_load(layers=orphaned_layers, avail_mem=avail_mem, avail_vram=avail_vram):
                                self.chain.repair(orphaned_layers, successor_2_data, succ_was_tail, replacement_node_id=backup_id)
                                
                                try:
                                    channel = grpc.insecure_channel(backup_info.get("address"))
                                    backup_stub = nodeservice_pb2_grpc.NodeServiceStub(channel)
                                    backup_stub.LoadLayers(nodeservice_pb2.Empty())
                                
                                    self._connect_to_successor()
                                    return    
                                except grpc.RpcError as e:
                                    logger.error(f"A gRPC error occurred while connecting to {backup_info.get('address')}: {e.code().name}")
                # else:
                logger.info("No backup nodes found. Setting this node as the tail...") 
                self.chain.update_chain_tail(self.chain.node_id)
                self.chain.update_successor(new_successor_data=None)
                self.chain.update_all_layer_loaded()

    def _reload_llm(self, layers: Tuple[int, int]):
        """Helper to reload the model with explicit GC"""
        logger.info(f"Reloading model with layers: {layers}")
        self.model = None
        self.llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.llm = LLM.load(self.model_path, is_client=False, layers_to_load=layers)
        self.model = self.llm.model
        self.num_local_layers = self.llm.model.num_layers
        self._update_memory_usage()

    def _connect_to_successor(self):
        """Establishes a gRPC connection to the successor node."""
        if self.successor_stub is not None:
            return

        if self.chain.is_tail() or self.chain.is_backup():
            self.successor_stub = None
            return

        successor_addr = self.chain.get_successor_address()
        if not successor_addr:
            logger.warning("Successor address not found.")
            self.successor_stub = None
            return

        try:
            channel = grpc.insecure_channel(successor_addr)
            grpc.channel_ready_future(channel).result(timeout=10)
            self.successor_stub = nodeservice_pb2_grpc.NodeServiceStub(channel)
            logger.info(f"Connection to successor: {successor_addr} established.")
        except grpc.FutureTimeoutError:
            logger.error(f"Connection to {successor_addr} timed out.")
            self.successor_stub = None
        except grpc.RpcError as e:
            logger.error(f"A gRPC error occurred while connecting to {successor_addr}: {e.code().name}")
            self.successor_stub = None


GRPC_PORT = 5001
UDP_PORT = 9999
MAX_MSG_SIZE = 100 * 1024 * 1024  # 100 MB


def serve():
    """The main function to start the server."""
    load_dotenv()
    
    # Read configuration from environment variables
    model_path_str = os.getenv("MODEL_PATH")
    model_path = Path(model_path_str)

    num_layers_str = os.getenv("NUM_LAYERS")
    num_layers = int(os.getenv("NUM_LAYERS")) if num_layers_str is not None else None

    added_delay_str = os.getenv("ADDED_DELAY")
    added_delay = float(os.getenv("ADDED_DELAY")) if added_delay_str is not None else None

    # hostname = socket.gethostname()
    my_ip = os.getenv("IP")
    if not my_ip:
        my_ip = get_ip_address()
    grpc_addr = f"{my_ip}:{GRPC_PORT}"

    # host_maddrs = os.getenv("HOST_MADDRS")
    host_maddrs = f"/ip4/{my_ip}/tcp/0"

    bootstrap_node_addr_str = os.getenv("BOOTSTRAP_NODE_ADDR")

    if not bootstrap_node_addr_str:
        bootstrap_node_addr_str = discover_bootstrap_node_address()

    if not bootstrap_node_addr_str:
        logger.error("Failed to find bootstrap node address.")

    # Connect to the bootstrap node to get its p2p Multiaddress
    if bootstrap_node_addr_str:
        peer_addr = get_bootstrap_peer_address(bootstrap_node_addr_str, attempts=15)

        if not peer_addr:
            return

        initial_peers = [peer_addr]
        logger.info(f"Successfully discovered bootstrap peer: {initial_peers[0]}")

    else:
        initial_peers = None  # Head server

    server_node = Server(
        model_path=model_path,
        num_layers=num_layers,
        added_delay=added_delay,
        host_maddrs=[host_maddrs],
        initial_peers=initial_peers,
        grpc_addr=grpc_addr,
    )

    # Start GRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2),
        options=[
            ("grpc.max_send_message_length", MAX_MSG_SIZE),
            ("grpc.max_receive_message_length", MAX_MSG_SIZE),
        ],
    )
    nodeservice_pb2_grpc.add_NodeServiceServicer_to_server(NodeServicer(server_node), server)
    server.add_insecure_port(grpc_addr)

    server.start()
    logger.info("Server is ready to accept grpc connections.")

    # Initialize the successor stub
    server_node._connect_to_successor()

    def _dht_heartbeat_task(server_node: Server):
        """Background task to keep DHT keys alive."""
        while True:
            time.sleep(HEARTBEAT_INTERVAL_S)
            server_node.chain.republish_keys()
            server_node._update_memory_usage()
            server_node.chain.update_all_layer_loaded()

    def _grpc_heartbeat_task(server_node: Server):
        """Backgroud task to check on the node's successor status"""
        while True:
            time.sleep(HEARTBEAT_INTERVAL_S)
            if server_node.chain.is_backup() or server_node.chain.is_tail():
                continue

            # Perform the health check on the successor
            try:
                server_node._connect_to_successor()
                if server_node.successor_stub is not None:
                    server_node.successor_stub.Check(nodeservice_pb2.Empty(), timeout=2)
                    logger.info(f"Successor is ALIVE.")  # Debugging
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE or e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    logger.warning(f"Successor failure detected during HEARTBEAT CHECK.")  # Debugging
                    server_node.chain.dht.store(ALL_LAYERS_KEY, False, EXPIRATION_S)
                    server_node.repair_chain()
                else:
                    logger.warning(f"A gRPC error occurred during health check: {e.code().name}")
                    server_node.successor_stub = None

    def _udp_discovery_server():
        """Background task that listens for bootstrap discovery requests and responds with the servers grpc address"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", UDP_PORT))

        response_b = grpc_addr.encode()

        while True:
            data, addr = sock.recvfrom(1024)
            if data == b"DISCOVER_BOOTSTRAP":
                sock.sendto(response_b, addr)

    grpc_heartbeat_thread = threading.Thread(target=_grpc_heartbeat_task, args=(server_node,), daemon=True)
    grpc_heartbeat_thread.start()

    dht_heartbeat_thread = threading.Thread(target=_dht_heartbeat_task, args=(server_node,), daemon=True)
    dht_heartbeat_thread.start()

    udp_discovery_thread = threading.Thread(target=_udp_discovery_server, daemon=True)
    udp_discovery_thread.start()

    # Shutdown handler
    def _handle_shutdown(signum, frame):
        server_node.dht.shutdown()
        server.stop(grace=2)

    signal.signal(signal.SIGINT, _handle_shutdown)  # Ctrl+C
    signal.signal(signal.SIGTERM, _handle_shutdown)  # docker stop

    server_node._load_llm()

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
