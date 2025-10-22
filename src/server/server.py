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

from core.llm_loader import LLM
from core.remote.utils import get_bootstrap_peer_address
from core.remote import nodeservice_pb2, nodeservice_pb2_grpc
from core.remote.serialization import *
from core.p2p.dht_manager import DHTManager
from core.p2p.chain_manager import ChainManager, HEARTBEAT_INTERVAL_S

logger = logging.getLogger(__name__)


class NodeServicer(nodeservice_pb2_grpc.NodeServiceServicer):
    def __init__(self, server_node: "Server"):
        self.server_node = server_node

    def GetPeerMultiaddr(self, request, context):
        visible_maddrs = self.server_node.dht.get_visible_maddrs()
        if not visible_maddrs:
            context.abort(grpc.StatusCode.UNAVAILABLE, "P2P address not yet available.")

        return nodeservice_pb2.MultiaddrResponse(multiaddr=str(visible_maddrs[1]))

    def RunLayers(self, request, context):
        # Create the successor stub if it doesn't exist
        if self.server_node.successor_stub is None:
            if self.server_node.chain.is_tail():
                self.server_node.successor_stub = None
            else:
                successor_addr = self.server_node.chain.get_successor_address()
                if not successor_addr:
                    context.abort(grpc.StatusCode.INTERNAL, "Successor not found for a non-tail node.")
                    return nodeservice_pb2.InferenceResponse()

                try:
                    channel = grpc.insecure_channel(successor_addr)
                    grpc.channel_ready_future(channel).result(timeout=10)

                    self.server_node.successor_stub = nodeservice_pb2_grpc.NodeServiceStub(channel)
                    logger.info(f"Connection to successor: {successor_addr} established.")  # Debugging
                except grpc.FutureTimeoutError:
                    logger.warning(f"Connection to {successor_addr} timed out.")
                except grpc.RpcError as e:
                    logger.error(f"A gRPC error occurred while connecting: {e.code().name}")

        # Deserialize the incoming request to a tensor
        input_tensor = message_to_tensor(request)

        # Extract metadata
        max_returned_tokens = request.max_returned_tokens
        seq_length = request.seq_length if request.HasField("seq_length") else None
        input_pos_val = request.input_pos if request.HasField("input_pos") else None
        input_pos = torch.tensor([input_pos_val]) if input_pos_val is not None else None

        # Run the actual inference logic
        final_layer_response = self.server_node.run_local_layers(input_tensor, max_returned_tokens, seq_length, input_pos)

        # Serialize the output tensor into a response
        return nodeservice_pb2.InferenceResponse(
            tensor_data=final_layer_response.tensor_data,
            tensor_shape=final_layer_response.tensor_shape,
            dtype=final_layer_response.dtype,
        )


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

        # Memory
        self.process = psutil.Process(os.getpid())
        self.memory_usage_mb = 0.0
        self.memory_limit_mb = self._get_container_memory_limit_mb()
        self.available_memory_mb = 0.0
        self._update_memory_usage()

        # Get the conig file
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # Calculate number of Transformer Layers to load
        if not num_layers:
            num_layers = self._layers_to_load(config, bytes_per_param=4)
            logger.info(f"Node with {self.available_memory_mb} MB available can load {num_layers} layers.")

        # Join the inference chain
        server_info = {"address": grpc_addr}

        self.chain.join_chain(
            server_info,
            num_layers=num_layers,
            num_total_layers=config["num_hidden_layers"],
        )

        # If all layers are loaded then this node acts as a backup and doesn't load any layers
        if self.chain.is_backup():
            return

        layers_to_loaded = self.chain.get_layers_loaded()

        # num_layers = self.chain.get_num_layers()

        self.llm = LLM.load(
            model_path,
            is_client=False,
            # num_layers=num_layers,
            # layers_start_idx=layers_loaded[0],
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

    def _layers_to_load(self, config: Dict[str, Any], bytes_per_param: int) -> int:
        """
        Determine the number of layers to load based the in memory size of a Transformer layer and the available memory of the server.

        Greedy Layer Allocation: Load as many layers as memory allows
        """

        total_layer_params = config["total_transformer_layer_params"]

        single_layer_memory_size = total_layer_params * bytes_per_param  # Bytes

        available_memory_bytes = self.available_memory_mb * 1024 * 1024

        layers_to_load = int(available_memory_bytes // single_layer_memory_size)

        if layers_to_load > config["num_hidden_layers"]:
            layers_to_load = config["num_hidden_layers"]

        return layers_to_load

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
    ) -> Union[nodeservice_pb2.InferenceRequest, nodeservice_pb2.InferenceResponse]:
        """
        This function runs inference on the server's assigned transformer layers and send the output to the next node.
        """
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
            self.layers_per_second = self.num_local_layers / self.computational_delay
            # print(f"Layers {self.llm.layers_loaded}: " f"Delay: {self.computational_delay:.4f}s " f"Layers/sec: {self.layers_per_second:.2f} ")

        if self.chain.is_tail():
            return tensor_to_response(h)

        # Call the successor via gRPC
        request = tensor_to_request(
            h,
            max_returned_tokens=max_returned_tokens,
            seq_length=seq_length,
            input_pos=input_pos.item() if input_pos is not None else None,
        )
        final_layer_response = self.successor_stub.RunLayers(request)
        return final_layer_response

GRPC_PORT = 5001

def serve():
    """The main function to start the server."""
    # Read configuration from environment variables
    model_path_str = os.getenv("MODEL_PATH")
    model_path = Path(model_path_str)

    num_layers_str = os.getenv("NUM_LAYERS")
    num_layers = int(os.getenv("NUM_LAYERS")) if num_layers_str is not None else None

    hostname = socket.gethostname()
    grpc_addr = f"{hostname}:{GRPC_PORT}"

    host_maddrs_str = os.getenv("HOST_MADDRS")
    host_maddrs = [host_maddrs_str]

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
        host_maddrs=host_maddrs,
        initial_peers=initial_peers,
        grpc_addr=grpc_addr,
    )

    def _heartbeat_task(server_node: Server):
        """Background task to keep DHT keys alive."""
        while True:
            time.sleep(HEARTBEAT_INTERVAL_S)
            server_node.chain.republish_keys()

    heartbeat_thread = threading.Thread(target=_heartbeat_task, args=(server_node,), daemon=True)
    heartbeat_thread.start()

    # Start GRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    nodeservice_pb2_grpc.add_NodeServiceServicer_to_server(NodeServicer(server_node), server)
    server.add_insecure_port(grpc_addr)

    # Shutdown handler
    def _handle_shutdown(signum, frame):
        server.stop(5)

    signal.signal(signal.SIGINT, _handle_shutdown)  # Ctrl+C
    signal.signal(signal.SIGTERM, _handle_shutdown)  # docker stop

    server.start()
    logger.info("Server is ready to accept grpc connections.")

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
