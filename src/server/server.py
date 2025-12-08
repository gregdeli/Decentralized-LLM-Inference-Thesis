"""Main server application logic"""

import json
import logging
import os
import signal
import threading
import socket
import time
from concurrent import futures
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import grpc
import psutil
import torch

from servicer import NodeServicer
from core.llm_loader import LLM
from core.remote.utils import get_bootstrap_peer_address
from core.remote import nodeservice_pb2, nodeservice_pb2_grpc
from core.remote.serialization import *
from core.p2p.dht_manager import DHTManager
from core.p2p.chain_manager import ChainManager, HEARTBEAT_INTERVAL_S, ALL_LAYERS_KEY, EXPIRATION_S

logger = logging.getLogger(__name__)


class Server:
    def __init__(
        self,
        model_path: Path,
        num_layers: int = None,
        time_it: bool = False,
        host_maddrs: List[str] = ["/ip4/0.0.0.0/tcp/4001"],
        initial_peers: List[str] = None,
        grpc_addr: str = "head-server:5001",
    ) -> None:
        # Create a DHT Node for the server
        self.dht = DHTManager(host_maddrs=host_maddrs, initial_peers=initial_peers)
        self.dht.start()
        self.chain = ChainManager(self.dht)

        self._repair_lock = threading.Lock()

        # Memory
        self.process = psutil.Process(os.getpid())
        self.memory_usage_mb = 0.0
        self.memory_limit_mb = self._get_container_memory_limit_mb()
        self.available_memory_mb = 0.0
        self._update_memory_usage()

        self.model_path = model_path

        # Get the config file
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        self.config = config

        # Calculate number of Transformer Layers to load
        if not num_layers:
            num_layers = self._mem_to_num_layers(bytes_per_param=4)
            logger.info(f"Node with {self.available_memory_mb} MB available can load {num_layers} layers.")

        # Join the inference chain
        server_info = {"address": grpc_addr}

        self.chain.join_chain(
            server_info,
            max_num_layers=num_layers,
            num_total_layers=config["num_hidden_layers"],
        )

        # If all layers are loaded then this node acts as a backup and doesn't load any layers
        if self.chain.is_backup():
            return

        layers_to_loaded = self.chain.get_layers_loaded()

        self.llm = LLM.load(
            model_path,
            is_client=False,
            layers_to_load=layers_to_loaded,
            time_it=time_it,
        )
        self.model = self.llm.model

        # Create successor stub
        self.successor_stub = None

        # Processing Rate
        self.num_local_layers = self.llm.model.num_layers
        self.layers_per_second = 0.0
        self.computational_delay = 0.0

    def _mem_to_num_layers(self, bytes_per_param: int) -> int:
        """
        Determine the number of layers that can be loaded load based the in memory size
        of a Transformer layer and the available memory of the server node.

        Greedy Layer Allocation: Load as many layers as memory allows
        """
        self._update_memory_usage()

        total_layer_params = self.config["total_transformer_layer_params"]

        single_layer_memory_size = total_layer_params * bytes_per_param  # Bytes

        # Thelei kai kapoio overhead logika
        # reserved_overhead = 500 * 1024 * 1024

        available_memory_bytes = self.available_memory_mb * 1024 * 1024

        max_num_layers = int(available_memory_bytes // single_layer_memory_size)

        if max_num_layers > self.config["num_hidden_layers"]:
            max_num_layers = self.config["num_hidden_layers"]

        return max_num_layers

    def _get_container_memory_limit_mb(self) -> Optional[float]:
        """Reads the container's memory limit from cgroup files."""
        cgroup_v2_path = "/sys/fs/cgroup/memory.max"

        limit_bytes = None
        with open(cgroup_v2_path, "r") as f:
            content = f.read().strip()
            if content != "max":
                limit_bytes = int(content)

        if limit_bytes:
            return limit_bytes / (1024 * 1024)
        return None

    def _update_memory_usage(self):
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

    def _print_mem_usage(self, title: str):
        logger.info(f"{title}: Memory Usage: {self.memory_usage_mb:.2f}/{self.memory_limit_mb}MB")
        logger.info(f"{title}: Available Memory: {self.available_memory_mb:.2f}MB")

    @torch.no_grad()
    def run_local_layers(
        self,
        input_tensor: torch.Tensor,
        max_returned_tokens: int,
        seq_length: int = None,
        input_pos: torch.Tensor = None,
        incoming_partial_rate: float = 0.0,
    ) -> Union[nodeservice_pb2.InferenceRequest, nodeservice_pb2.InferenceResponse]:
        """
        This function runs inference on the server's assigned transformer layers and send the output to the next node.
        """
        # Ensure inputs are on the same device as the model
        device = self.llm.device

        if input_tensor.device != device:
            input_tensor = input_tensor.to(device)

        if input_pos is not None and input_pos.device != device:
            input_pos = input_pos.to(device)

        # KV Cache
        if not self.llm.kv_cache_initialized:
            self._update_memory_usage()
            # self._print_mem_usage("After Loading Model")
            device = self.llm.preprocessor.device
            # Na allaksw to batch_size otan kanw batched inference
            self.model.set_kv_cache(batch_size=1, max_seq_length=max_returned_tokens, device=device)
            self.llm.kv_cache_initialized = True
            self._update_memory_usage()
            # self._print_mem_usage("After Setting KV Cache")

        # Dynamically grow the kv cache size if necessary
        elif self.llm.prev_generated_seq_length < max_returned_tokens:
            tmp_device = self.model.mask_cache.device
            self.model.clear_kv_cache()
            self.model.set_kv_cache(batch_size=1, max_seq_length=max_returned_tokens, device=tmp_device)
            self._update_memory_usage()
            # self._print_mem_usage("After Growing KV Cache")

        self.llm.prev_generated_seq_length = max_returned_tokens

        start_time = time.perf_counter()

        # logger.info(f"Processing layers {self.llm.layers_loaded}...")  # Debugging

        h = self.model.forward_server(input_tensor, seq_length, input_pos)

        end_time = time.perf_counter()
        self.computational_delay = end_time - start_time

        # Get the server's layer processing rate for this inference run
        if self.computational_delay > 0 and self.num_local_layers > 0:
            current_rate = self.num_local_layers / self.computational_delay

            # Moving average to avoid jitter
            if self.layers_per_second == 0:
                self.layers_per_second = current_rate
            else:
                self.layers_per_second = (0.7 * self.layers_per_second) + (0.3 * current_rate)

        logger.info(f"Layers {self.llm.layers_loaded}: " f"Delay: {self.computational_delay:.4f}s " f"Layers/sec: {self.layers_per_second:.2f} ")

        # Calculate partial rate to send forward
        my_partial_rate = incoming_partial_rate + self.layers_per_second

        # Move output tensor back to the cpu for serialization
        h = h.cpu()  # is this neccessary?

        if self.chain.is_tail():
            response = tensor_to_response(h)
            response.total_rate = my_partial_rate
            return response

        # Call the successor via gRPC
        request = tensor_to_request(
            h,
            max_returned_tokens=max_returned_tokens,
            seq_length=seq_length,
            input_pos=input_pos.item() if input_pos is not None else None,
        )

        request.partial_rate = my_partial_rate

        try:
            final_layer_response = self.successor_stub.RunLayers(request, timeout=4)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE or e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                logger.warning(f"Successor failure detected during INFERENCE.")  # Debugging
                dead_successor_data = self.chain.get_failed_successor_data()
                if dead_successor_data:
                    self.chain.dht.store(ALL_LAYERS_KEY, False, EXPIRATION_S)
                    # self.repair_chain(dead_successor_data)
                    self.repair_chain()
                    return nodeservice_pb2.InferenceResponse(error_message="A node in the chain failed. The chain is being repaired...")
                else:
                    logger.error(f"Could not retrieve successor data from DHT! Chain is broken.")
                    return nodeservice_pb2.InferenceResponse(
                        error_message="A node in the chain failed and its data could not be retrieved. Chain is broken."
                    )

        return final_layer_response

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
            ideal_layer_count = num_total_layers * (self.layers_per_second / total_system_rate)
        else:
            ideal_layer_count = 0

        # Rounding logic
        if self.chain.is_tail():
            # The tail takes whatever is left
            my_layer_count = num_total_layers - start_layer_index
        else:
            my_layer_count = int(round(ideal_layer_count))
            if my_layer_count < 1:
                my_layer_count = 1
            if (start_layer_index + my_layer_count) >= num_total_layers:
                # Leave at least 1 layer for the rest of the chain if not tail
                my_layer_count = num_total_layers - start_layer_index - 1

        end_layer_index = start_layer_index + my_layer_count - 1
        new_layers = (start_layer_index, end_layer_index)

        # Load Layers
        if new_layers != self.llm.layers_loaded:
            logger.info(f"Reloading model with new layers: {new_layers} (Previous: {self.llm.layers_loaded})")

            # Clear GPU mem before reloading
            if torch.cuda.is_available():
                self.model = None
                self.llm = None
                torch.cuda.empty_cache()

            self.llm = LLM.load(model_path=self.model_path, is_client=False, layers_to_load=new_layers)
            self.model = self.llm.model
            self.num_local_layers = my_layer_count

            # Update Chain info
            self.chain.update_layers_loaded(new_layers)
        else:
            logger.info("Layer assignment unchanged.")

        # Propagate to successor
        if not self.chain.is_tail() and self.successor_stub:
            next_start_index = end_layer_index + 1

            request = nodeservice_pb2.ReallocateRequest(total_rate=total_system_rate, start_layer_index=next_start_index)
            try:
                self.successor_stub.Reallocate(request)
            except grpc.RpcError as e:
                logger.error(f"Failed to propagate Reallocation to successor: {e}")

    def _connect_to_successor(self):
        """Establishes a gRPC connection to the successor node."""
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

    def repair_chain(self):  # , dead_successor_data: Dict[str, Any]):
        """
        Method that repairs the inference chain after detecting this node's successor is dead.
        To repair the chain:
        1. If this node can hold the orphaned layers of the failed node, it loads them.
        2. Else, if a backup node is available it prompts it to take the place of the failed node.
        3. Otherwise, the chain can't be repaired.
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
            orphaned_layers = dead_successor_data.get("layers_loaded")
            successor_2_data = dead_successor_data.get("successor")
            succ_was_tail = dead_successor_data.get("was_tail")
            logger.info(
                f"Attempting to repair chain. Orphaned layers: {orphaned_layers}, Successor 2 Data: {successor_2_data}, Was TAIL: {succ_was_tail}"
            )

            # Check if this node has enough memory to load the orphaned layers
            if self._can_load(orphaned_layers):
                layers_to_load = (self.llm.layers_loaded[0], orphaned_layers[1])

                self.llm = LLM.load(model_path=self.model_path, is_client=False, layers_to_load=layers_to_load)
                self.model = self.llm.model

                # Update the DHT with this nodes new info
                self.chain.repair(layers_to_load, successor_2_data, succ_was_tail)

                # Recreate successor stub to the new successor
                self._connect_to_successor()

            # Else, try to find a backup node to take over
            else:
                logger.info("Cannot load layers. Searching for a backup node...")
                # TODO: Implement backup node discovery
                # backup_node = self.chain.find_backup_node()
                # if backup_node:
                #    logger.info(f"Found backup node {backup_node.id}. Triggering takeover...")
                #    # TODO: Implement gRPC call to backup node to tell it to
                #    # take over the 'dead_successor_data'
                #    # This node would then set its successor to the backup node.
                #    return

        # Otherwise, the chain can't be repaired

    def _can_load(self, orphaned_layers: Tuple[int, int]) -> bool:
        """Checks if this node can load the orphaned layers of a failed successor"""
        max_num_layers = self._mem_to_num_layers(bytes_per_param=4)
        if max_num_layers <= 0:
            return False

        num_orphaned_layers = orphaned_layers[1] - orphaned_layers[0] + 1
        if max_num_layers < num_orphaned_layers:
            return False

        return True


GRPC_PORT = 5001
UDP_PORT = 9999


def serve():
    """The main function to start the server."""
    # Read configuration from environment variables
    model_path_str = os.getenv("MODEL_PATH")
    model_path = Path(model_path_str)

    num_layers_str = os.getenv("NUM_LAYERS")
    num_layers = int(os.getenv("NUM_LAYERS")) if num_layers_str is not None else None

    hostname = socket.gethostname()
    grpc_addr = f"{hostname}:{GRPC_PORT}"

    host_maddrs = os.getenv("HOST_MADDRS")

    bootstrap_node_addr_str = os.getenv("BOOTSTRAP_NODE_ADDR")

    # Connect to the bootstrap node to get its p2p Multiaddress
    if bootstrap_node_addr_str:
        peer_addr = get_bootstrap_peer_address(bootstrap_node_addr_str, attempts=10)

        if not peer_addr:
            return

        initial_peers = [peer_addr]
        logger.info(f"Successfully discovered bootstrap peer: {initial_peers[0]}")

    else:
        initial_peers = None  # Head server

    server_node = Server(
        model_path=model_path,
        num_layers=num_layers,
        host_maddrs=[host_maddrs],
        initial_peers=initial_peers,
        grpc_addr=grpc_addr,
    )

    # Start GRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
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

    def _grpc_heartbeat_task(server_node: Server):
        """Backgroud task to check on the node's successor status"""
        while True:
            time.sleep(HEARTBEAT_INTERVAL_S)
            if server_node.chain.is_backup() or server_node.chain.is_tail():
                continue

            # Perform the health check on the successor
            try:
                if server_node.successor_stub is not None:
                    server_node.successor_stub.Check(nodeservice_pb2.Empty(), timeout=2)
                    logger.info(f"Successor is ALIVE.")  # Debugging
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE or e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    logger.warning(f"Successor failure detected during HEARTBEAT CHECK.")  # Debugging
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
        server.stop(5)

    signal.signal(signal.SIGINT, _handle_shutdown)  # Ctrl+C
    signal.signal(signal.SIGTERM, _handle_shutdown)  # docker stop

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
