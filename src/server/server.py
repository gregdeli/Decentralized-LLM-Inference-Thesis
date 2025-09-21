"""Main server application logic"""

import grpc
import os
from concurrent import futures
import torch
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple
import time
import signal
import psutil
import socket

from core.llm_loader import LLM
from core.remote import inference_pb2, inference_pb2_grpc
from core.remote.serialization import *
from core.p2p.dht_manager import DHTManager
from core.p2p.chain_manager import ChainManager


class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def __init__(self, server_node: "Server"):
        self.server_node = server_node

    def RunLayers(self, request, context):
        # Create the successor stub if it doesn't exist
        if self.server_node.successor_stub is None:
            # Get this nodes successor address from the DHT
            successor_addr = self.server_node.chain.get_successor_address()

            if successor_addr is not None:
                channel = grpc.insecure_channel(successor_addr)
                self.successor_stub = inference_pb2_grpc.InferenceStub(channel)
            else:
                context.abort(grpc.StatusCode.INTERNAL, "Successor not found for a non-tail node.")
                return inference_pb2.InferenceResponse()

        # Deserialize the incoming request to a tensor
        input_tensor = response_to_tensor(request)

        # Extract metadata
        max_returned_tokens = request.max_returned_tokens
        seq_length = request.seq_length if request.HasField("seq_length") else None
        input_pos_val = request.input_pos if request.HasField("input_pos") else None
        input_pos = torch.tensor([input_pos_val]) if input_pos_val is not None else None

        # Run the actual inference logic
        final_layer_response = self.server_node.run_local_layers(input_tensor, max_returned_tokens, seq_length, input_pos)

        # Serialize the output tensor into a response
        return inference_pb2.InferenceResponse(
            tensor_data=final_layer_response.tensor_data,
            tensor_shape=final_layer_response.tensor_shape,
            dtype=final_layer_response.dtype,
        )


class Server:
    def __init__(
        self,
        model_path: Path,
        num_layers: int = None,
        # layers_start_idx: int = 0,
        # is_head: bool = False,
        # successor_addr: Optional[str] = None,
        time_it: bool = False,
        host_maddrs: str = "/ip4/0.0.0.0/tcp/4001",
        initial_peers: str = None,
        grpc_port: str = 5001,
    ) -> None:
        # Create a DHT Node for the server
        self.dht = DHTManager(host_maddrs=[host_maddrs], initial_peers=initial_peers)
        self.dht.start()
        self.chain = ChainManager(self.dht)

        # Join the inference chain
        hostname = socket.gethostname()
        server_info = {"num_layers": num_layers, "address": f"{hostname}:{grpc_port}"}
        self.chain.join_chain(server_info)

        layers_loaded = self.chain.get_layers_loaded()

        self.llm = LLM.load(
            model_path,
            is_client=False,
            num_layers=num_layers,
            layers_start_idx=layers_loaded[0],
            time_it=time_it,
        )
        self.model = self.llm.model
        # self.is_head = is_head
        # self.is_tail = self.llm.layers_loaded[1] == (self.llm.config["num_hidden_layers"] - 1)

        # self.successor_addr = successor_addr

        # Create successor stub
        self.successor_stub = None
        # if self.successor_addr:
        #     channel = grpc.insecure_channel(self.successor_addr)
        #     self.successor_stub = inference_pb2_grpc.InferenceStub(channel)

        # Processing Rate
        self.num_local_layers = self.llm.num_layers
        self.layers_per_second = 0.0
        self.computational_delay = 0.0

        # Memory
        self.process = psutil.Process(os.getpid())
        self.memory_usage_mb = 0.0
        self.memory_limit_mb = self._get_container_memory_limit_mb()
        self.available_memory_mb = 0.0
        self._update_memory_usage()

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
        print(f"{title}: Memory Usage: {self.memory_usage_mb:.2f}MB  Available Memory: {self.available_memory_mb:.2f}MB")

    @torch.no_grad()
    def run_local_layers(
        self,
        input_tensor: torch.Tensor,
        max_returned_tokens: int,
        seq_length: int = None,
        input_pos: torch.Tensor = None,
    ) -> Union[inference_pb2.InferenceRequest, inference_pb2.InferenceResponse]:
        """
        This function runs inference on the server's assigned transformer layers and send the output to the next node.
        """
        if not self.llm.kv_cache_initialized:
            self._update_memory_usage()
            self._print_mem_usage("After Loading Model")
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


def serve():
    """The main function to start the server."""
    # Read configuration from environment variables
    model_path_str = os.getenv("MODEL_PATH")
    model_path = Path(model_path_str)
    num_layers = int(os.getenv("NUM_LAYERS"))
    # layers_start_idx = int(os.getenv("LAYERS_START_IDX"))
    # is_head_str = os.getenv("IS_HEAD", "False")
    # is_head = is_head_str.lower() in ("true", "1")
    # successor_addr = os.getenv("SUCCESSOR_ADDR", None)
    grpc_port = os.getenv("GRPC_PORT", "5001")
    host_maddrs = os.getenv("HOST_MADDRS")
    initial_peers = os.getenv("INITIAL_PEERS")

    server_node = Server(
        model_path=model_path,
        num_layers=num_layers,
        # layers_start_idx=layers_start_idx,
        # is_head=is_head,
        # successor_addr=successor_addr,
        host_maddrs=host_maddrs,
        initial_peers=initial_peers,
        grpc_port=grpc_port,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(server_node), server)
    server.add_insecure_port(f"[::]:{grpc_port}")

    # Shutdown handler
    def _handle_shutdown(signum, frame):
        server.stop(5)

    signal.signal(signal.SIGINT, _handle_shutdown)  # Ctrl+C
    signal.signal(signal.SIGTERM, _handle_shutdown)  # docker stop

    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
