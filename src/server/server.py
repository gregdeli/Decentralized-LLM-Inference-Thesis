"""Main server application logic"""

import json
import logging
import sys
import os
import signal
import threading
import socket
import random
from dotenv import load_dotenv
import gc
import ctypes
import time
from concurrent import futures
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import grpc
import psutil
import torch

# Add src/ to python paths or app/ if running in container
src_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_root))

from servicer import NodeServicer
from core.llm_loader import LLM
from core.constants import *
from core.utils import (
    get_dtype_from_config,
    calculate_transformer_params,
    calculate_final_output_params,
    update_model_config,
    can_load
)
from core.remote.utils import *
from core.remote import nodeservice_pb2, nodeservice_pb2_grpc
from core.remote.serialization import *
from core.p2p.dht_manager import DHTManager
from core.p2p.chain_manager import (
    ChainManager,
    ChainStatus,
    HEARTBEAT_INTERVAL_S,
    ALL_LAYERS_KEY,
    EXPIRATION_S,
    DIGITS_SHOW,
)

logger = logging.getLogger(__name__)


class Server:
    def __init__(
        self,
        grpc_addr: str,
        model_path: Path,
        num_layers: int = None,
        added_delay: float = None,  # Debugging
        time_it: bool = False,
        host_maddrs: List[str] = ["/ip4/0.0.0.0/tcp/4001"],
        initial_peers: List[str] = None,
    ) -> None:
        self.added_delay = added_delay

        self.grpc_addr = grpc_addr

        self.model_path = model_path
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        if torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
                free_mb = free_bytes / (1024 * 1024)
                
                # Force PyTorch's CUDACachingAllocator to initialize
                _dummy = torch.empty(1, device="cuda")

                free_bytes, _ = torch.cuda.mem_get_info()
                free_mb = free_bytes / (1024 * 1024)
                if free_mb >= RESERVED_MEM_MB:
                    self.device = "cuda"
            except RuntimeError as e:
                logger.warning(f"Failed to allocate CUDA context, defaulting to CPU. Error: {e}")
        
        # Locks
        self._repair_lock = threading.Lock()
        self._inference_lock = threading.Lock()

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

        # Initialize LLM State
        self.llm = None
        self.model = None
        self.num_local_params = 0
        self.output_layer_temporal_tle = 1 # Transformer Layer Equivilant 
        self.output_layer_memory_tle = 1
        self.layer_mem_size_mb = 0

        # Determine Model Parameter Load Capacity
        self._set_config_param_counts()

        # Determine Layer Load Capacity
        self._update_memory_usage(update_on_dht=False)
        if not num_layers:
            num_layers = self._mem_to_num_layers()
            available = (
                self.available_vram_mb if self.available_vram_mb else self.available_memory_mb
            )
            logger.info(f"Node with {available} MB available can load {num_layers} layers.")
        elif not self._can_load(num_layers=num_layers):
            logger.warning(
                f"Requested {num_layers} layers but memory is insufficient. Adjusting..."
            )
            num_layers = self._mem_to_num_layers()


        # Join the Inference Chain
        server_info = {
            "id": self.chain.node_id,
            "address": grpc_addr,
            "hostname": socket.gethostname(),
            "processing_rate": 0.0,
            "is_client": False
        }

        # Become the chain leader to join the chain 
        # self.chain.become_chain_leader()

        self.chain.join_chain(
            server_info,
            max_num_layers=num_layers,
            output_layer_memory_tle=self.output_layer_memory_tle,
            num_total_layers=self.config.get("num_hidden_layers"),
            num_total_params=self.config.get("num_total_params"),
        )
        self.chain.update_output_layer_memory_tle(self.output_layer_memory_tle)

        # Metrics 
        self.inference_delay = 0.0
        self.processing_rate = 0
        self.grpc_overhead = 0.0
        # self.succ_network_latency = 0.0

        # Successor Stub
        self.successor_stub = None
        self.successor_stub_addr = None  # Address that corresponds to the successor_stub. To check if same with dht succ. addr.

        # Client Stub
        self.client_stub = None
        self.client_stub_addr = None
    def _load_llm(
        self,
        layers_to_load: Optional[Tuple[int, int]] = None,
        load_output_layer: Optional[bool] = None,
        update_on_dht: bool = True,
        time_it: bool = False,
    ):
        """Loads the layers assigned by the ChainManager."""
        if layers_to_load is None and load_output_layer is None:
            layers_to_load = self.chain.get_layers()
            load_output_layer = self.chain.get_load_output_layer()

        logger.info(f"Loading layers: {layers_to_load}...")
        if load_output_layer:
            logger.info("Loading Final Output Layer...") 

        self.llm = LLM.load(
            self.model_path,
            device=self.device,
            load_initial_layer=False,
            layers_to_load=layers_to_load,
            load_output_layer=load_output_layer,
            time_it=time_it,
        )
        self.model = self.llm.model

        layer_params = self.model.num_layers * self.config.get("transformer_layer_params")
        output_params = self.config.get("final_output_params") if self.llm.output_layer_loaded else 0
        self.num_local_params = layer_params + output_params

        # Update DHT and Chain Status
        self.chain.update_device(self.llm.device)
        self._update_memory_usage()
        # if not self.chain.is_backup() and update_on_dht:
        if update_on_dht:
            self.chain.update_layers_loaded(self.model.num_layers>0)
            if self.llm.output_layer_loaded:
                self.chain.update_load_output_layer(False)
            self.chain.update_output_layer_loaded(self.llm.output_layer_loaded)
            self.chain.update_all_layers_loaded()

    def _unload_llm(self):
        """Helper to unload the model with explicit GC"""
        if self.model:
            self.model.clear_all_kv_caches()
        self.model = None
        self.llm = None
        self.num_local_params = 0

        # Release freed memory
        gc.collect()

        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception as e:
            logger.warning(f"Failed to trim memory: {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._update_memory_usage()
        # if update_on_dht:
        #     self.chain.update_layers(None)
        #     self.chain.update_layers_loaded(False)
        #     self.chain.update_output_layer_loaded(False)

    def _reload_llm(self, layers: Tuple[int, int], load_output_layer: bool = False):
        """Helper to reload the model with explicit GC"""
        logger.info(f"Reloading model with layers: {layers} | Load Output Layer: {load_output_layer}...")

        self._unload_llm()

        self.llm = LLM.load(
            self.model_path, 
            device=self.device,
            load_initial_layer=False, 
            layers_to_load=layers, 
            load_output_layer=load_output_layer
        )
        self.model = self.llm.model
        
        layer_params = self.model.num_layers * self.config.get("transformer_layer_params")
        output_params = self.config.get("final_output_params") if self.llm.output_layer_loaded else 0
        self.num_local_params = layer_params + output_params

        self._update_memory_usage()
        if self.llm.output_layer_loaded:
            self.chain.update_load_output_layer(False)
        self.chain.update_layers_loaded(self.model.num_layers>0)
        self.chain.update_output_layer_loaded(self.llm.output_layer_loaded)

    def _reload_config(self):
        with open(self.model_path / "config.json", "r") as f:
            self.config = json.load(f)

    def _set_config_param_counts(self) -> None:
        """
        Calculates the transformer_layer_params, final_output_params, and num_total_params of the model and updates the config file
        """
        self._update_memory_usage(update_on_dht=False)

        total_params = self.config.get("num_total_params")
        layer_params = self.config.get("transformer_layer_params")
        final_output_params = self.config.get("final_output_params")
        
        if total_params is None:
            if layer_params is None:
                layer_params = calculate_transformer_params(self.model_path)
                # update_config_layer_param_count(self.model_path, layer_params)
                update_model_config(self.model_path, "transformer_layer_params", layer_params)
                self._reload_config()
            
            if final_output_params is None:
                final_output_params = calculate_final_output_params(self.model_path)
                # update_config_final_output_param_count(self.model_path, final_output_params)
                update_model_config(self.model, "final_output_params", final_output_params)
                self._reload_config()
            
            total_params = self.config.get("num_hidden_layers") * layer_params + final_output_params
            # update_config_total_param_count(self.model_path, total_params)
            update_model_config(self.model_path, "num_total_params", total_params)
            self._reload_config()

        # Also set self.output_layer_memory_tle
        if "head_dim" not in self.config:
            self.config["head_dim"] = self.config["hidden_size"] // self.config["num_attention_heads"]

        param_dtype = get_dtype_from_config(self.config)
        bytes_per_param = param_dtype.itemsize

        layer_param_mem_size_mb = (layer_params * bytes_per_param) / (1024 * 1024)
        layer_kv_cache_mem_size_mb = (2 * self.config.get("num_key_value_heads") * GLOBAL_MAX_SEQ_LEN * self.config.get("head_dim") * bytes_per_param) / (1024 * 1024)
        self.layer_mem_size_mb = layer_param_mem_size_mb + layer_kv_cache_mem_size_mb
        if self.config.get("layer_mem_size_mb") is None:
            update_model_config(self.model_path, "layer_mem_size_mb", self.layer_mem_size_mb)
            self._reload_config()

        # Set the output layer's equivilance to a transformer layer in memory
        self.output_layer_memory_tle = (final_output_params * bytes_per_param) / (self.layer_mem_size_mb * 1024 * 1024)
        if self.config.get("output_layer_memory_tle") is None:
            update_model_config(self.model_path, "output_layer_memory_tle", self.output_layer_memory_tle)
            self._reload_config()


    def _mem_to_num_layers(
            self, 
            available_vram: Optional[float] = None,
            available_memory: Optional[float] = None,
            round_it: bool = True
        ) -> int:
        """
        Calculates how many transformer layers fit in the available Memory/VRAM taking into account the KV Cache 
        
        Arguements avail_vram, avail_memory are used in reallocation.
        """
        self._update_memory_usage(update_on_dht=False)

        available_vram = available_vram if available_vram is not None else self.available_vram_mb
        available_memory = available_memory if available_memory is not None else self.available_memory_mb
        
        if self.device == "cuda":
            available = available_vram - self.memory_usage_mb - RESERVED_MEM_MB
        else:
            available = available_memory - RESERVED_MEM_MB

        if available <= 0:
            return 0

        max_num_layers = available / self.layer_mem_size_mb
        if round_it:
            max_num_layers = int(round(max_num_layers))

        return min(max_num_layers, self.config.get("num_hidden_layers") + self.output_layer_memory_tle)

    def _can_load(
        self,
        num_layers: Optional[int] = None,
        layers: Optional[Tuple[int, int]] = None,
        output_layer: Optional[bool] = False,
    ) -> bool:
        """Checks if this node can load a certain number of layers or a range of layers"""
        # The requested additional number of layers
        requested_num_layers = num_layers if num_layers else (layers[1] - layers[0] + 1) if layers else 0

        # Calculate the memory used currently by layers that are loaded
        current_num_layers = self.model.num_layers
        if self.model.output_layer_loaded:
            current_num_layers += self.output_layer_memory_tle

        theoretical_mem_in_use = current_num_layers * self.layer_mem_size_mb + RESERVED_MEM_MB
        usage = self.vram_usage_mb if self.vram_usage_mb > 0 else self.memory_usage_mb

        mem_in_use = theoretical_mem_in_use if theoretical_mem_in_use > usage else usage

        true_available_memory = None
        true_available_vram = None
        if self.available_vram_mb > 0:
            true_available_vram = usage + self.available_vram_mb - mem_in_use
        else:
            true_available_memory = usage + self.available_memory_mb - mem_in_use

        # Calculate the extra number of layers that can be loaded
        max_extra_num_layers = self._mem_to_num_layers(
            available_vram=true_available_vram,
            available_memory=true_available_memory,
        )

        # Determine if the requested can be loaded
        if requested_num_layers > 0:
            if max_extra_num_layers < requested_num_layers:
                return False
            
            if output_layer:
                max_extra_num_layers -= requested_num_layers
                return max_extra_num_layers >= self.output_layer_memory_tle
            
            return True
        
        return max_extra_num_layers >= self.output_layer_memory_tle

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

        # KV Cache Size
        if hasattr(self, "model") and self.model is not None:
            kv_cache_memories = self.model.get_kv_cache_memory_sizes()
            if update_on_dht:
                self.chain.update_kv_cache_size(kv_cache_memories)

        if update_on_dht:
            self.chain.update_memory(
                self.memory_usage_mb, self.memory_limit_mb, self.available_memory_mb
            )

            if self.device == "cuda":
                self.chain.update_vram(
                    self.vram_usage_mb, self.vram_limit_mb, self.available_vram_mb
                )

    def _ensure_kv_cache(self, client_id: str) -> None:
        """Ensures the KV cache is initialized and large enough for the request."""
        # Initialize the client's kv cache if necessary
        if not self.model.client_has_cache(client_id):
            num_clients = self.chain.get_num_clients()
            max_seq_length = GLOBAL_MAX_SEQ_LEN // num_clients if num_clients else GLOBAL_MAX_SEQ_LEN

            self.model.add_client_cache(
                client_id, 
                batch_size=1,
                max_seq_length=max_seq_length,
                device=self.device,
                dtype=self.llm.dtype
            )

            self._update_memory_usage(update_on_dht=True)


    def _calculate_processing_rate(self, delay: float):
        """Calculates and updates the moving average of layers processed per second."""
        if delay <= 0:
            return

        num_layers = self.model.num_layers

        if self.model.output_layer_loaded:
            num_layers += self.output_layer_temporal_tle
        
        if num_layers == 0:
            return 
            
        current_rate = num_layers / delay

        # Moving average to avoid jitter
        if self.processing_rate == 0:
            self.processing_rate = current_rate
        else:
            self.processing_rate = (0.7 * self.processing_rate) + (0.3 * current_rate)

    @torch.no_grad()
    def _profile_node(
        self, 
        dummy_seq_length: int = 30, 
        profiling_runs: int = 100, 
        profiling_duration_s: int = 3
    ):
        """Measures a backup node's processing rate by doing a fake generation on a dummy input."""

        # time.sleep(random.uniform(0.5, 3.0))

        other_node_profiling =  self.chain.get_chain_status() == ChainStatus.PROFILING
        while other_node_profiling:
            logger.info(f"Another node is currently profiling. Sleeping for {profiling_duration_s} seconds...")
            time.sleep(profiling_duration_s)
            other_node_profiling = self.chain.get_chain_status() == ChainStatus.PROFILING

        # self.chain.become_chain_leader()
        self.chain.update_chain_status(ChainStatus.PROFILING)

        # ------ Transformer Layer Profiling ------

        # First profile a signle transformer layer to get the nodes processing rate and 
        # the transformer layers inference delay 
        self._load_llm(layers_to_load=(0,0), load_output_layer=False, update_on_dht=False)

        # Inference with dummy input
        hidden_size = self.config["hidden_size"]

        dummy_input = torch.randn(
            1, dummy_seq_length, hidden_size, device=self.device, dtype=self.llm.dtype
        )
        dummy_input_pos = None

        # KV Cache
        self._ensure_kv_cache(client_id="profiling")
        self.model.set_active_client(client_id="profiling")

        starting_dummy_seq_len = dummy_seq_length

        prof_start_time = time.perf_counter()
        for i in range(profiling_runs):
            start = time.perf_counter()
            _ = self.model.forward_server(
                dummy_input, seq_length=dummy_seq_length, input_pos=dummy_input_pos
            )
            if self.added_delay:
                time.sleep(self.added_delay)

            transformer_layer_delay = time.perf_counter() - start

            self._calculate_processing_rate(transformer_layer_delay)

            logger.info(
                f"Profiling: Layers {self.llm.layers_loaded}: "
                f"Delay: {transformer_layer_delay}s "
                f"Layers/sec: {self.processing_rate} "
            )

            dummy_input = torch.randn(1, 1, hidden_size, device=self.device, dtype=self.llm.dtype)
            current_pos = starting_dummy_seq_len + (i + 1)
            dummy_input_pos = torch.tensor([current_pos], device=self.device)
            dummy_seq_length = 1

            profiling_time = time.perf_counter() - prof_start_time
            if profiling_time >= profiling_duration_s:
                break
        
        self._unload_llm()

        # ---------- Output Layer Profiling ---------------
        self._load_llm(load_output_layer=True, update_on_dht=False)

        dummy_input = torch.randn(
            1, dummy_seq_length, hidden_size, device=self.device, dtype=self.llm.dtype
        )
        prof_start_time = time.perf_counter()
        for i in range(profiling_runs):
            start = time.perf_counter()
            _ = self.model.forward_server(dummy_input)
            if self.added_delay:
                time.sleep(self.added_delay)

            output_layer_delay = time.perf_counter() - start

            logger.info(
                f"Profiling: Output Layer: "
                f"Delay: {output_layer_delay}s "
            )

            profiling_time = time.perf_counter() - prof_start_time
            if profiling_time >= profiling_duration_s:
                break
        
        self._unload_llm()

        # Calculate the output layer's Transformer Layer Equivilant (TLE)
        self.output_layer_temporal_tle = output_layer_delay / transformer_layer_delay
        self.chain.update_output_layer_temporal_tle(self.output_layer_temporal_tle)

        self.chain.update_processing_rate(self.processing_rate)
        
        self._update_memory_usage()

        if self.chain.get_all_layers_loaded():
            self.chain.update_chain_status(ChainStatus.READY)
        else:    
            self.chain.update_chain_status(ChainStatus.UNREADY)

    @torch.no_grad()
    def run_local_layers(
        self,
        input_tensor: torch.Tensor,
        max_returned_tokens: int,
        seq_length: int = None,
        input_pos: torch.Tensor = None,
        response_address: str = None,
    ) -> nodeservice_pb2.InferenceResponse:
        """
        This function runs inference on the server's assigned transformer layers and send the output to the next node.
        """
        # Set chain status
        # self.chain.update_chain_status(ChainStatus.RUNNING)

        if not self.llm:
            return nodeservice_pb2.InferenceResponse(
                error_message="A node in the chain does not have its layers loaded."
            )

        # Ensure inputs are on the same device as the model
        device = self.llm.device

        if input_tensor.device != device:
            input_tensor = input_tensor.to(device)

        if input_pos is not None and input_pos.device != device:
            input_pos = input_pos.to(device)

        # If KV caches have been reallocated the generation needs to stop because the context is lost
        if self.model.client_cache_reallocated(client_id=response_address) and input_pos is not None:
            self._connect_to_client(response_address)
            self.client_stub.ReceiveResponse(
                nodeservice_pb2.InferenceResponse(error_message="KV Caches have been reallocated!")
            )
            return

        # Set KV Cache and Process Layers
        with self._inference_lock:
            self._ensure_kv_cache(client_id=response_address)

            self.model.set_active_client(client_id=response_address)

            # --- Inference ---
            start = time.perf_counter()

            logger.info(f"Processing Layers {self.llm.layers_loaded} | Output: {self.llm.output_layer_loaded} from ({response_address})...")

            h = self.model.forward_server(input_tensor, seq_length, input_pos)
            if self.added_delay:
                time.sleep(self.added_delay)
        
            self.inference_delay = time.perf_counter() - start
        
            self.chain.update_inference_delay(self.inference_delay)
            
            self._calculate_processing_rate(self.inference_delay)
            self.chain.update_processing_rate(self.processing_rate)

        # logger.info(
        #     f"Layers {self.llm.layers_loaded}: "
        #     f"Delay: {self.inference_delay:.4f}s "
        #     f"Rate: {self.processing_rate} params/sec"
        # )

        # Move output tensor back to the cpu for serialization
        h = h.cpu()

        # To prevent nonesense output when a node fails during inference
        # if not self.chain.get_all_layers_loaded():
        #     # self.chain.update_chain_status(ChainStatus.UNREADY)
        #     self._connect_to_client(response_address)
        #     self.client_stub.ReceiveResponse(
        #         nodeservice_pb2.InferenceResponse(error_message="Inference requested without all the layers being loaded!")
        #     )
        #     return

        # If TAIL Node -> Send response to Client
        if self.chain.is_tail() and self.chain.get_output_layer_loaded():
            logits = h
            next_token = self.llm.sample_logits(logits)

            # Benchmarking
            self.chain.update_chain_throughput()

            if not response_address:
                logger.error("Tail node has no response_address for the client!")
                return

            response = tensor_to_response(next_token)

            # Connect to Client
            try:
                self._connect_to_client(client_addr=response_address)

                logger.info(f"Sending Response to client ({self.client_stub_addr})...")

                start = time.perf_counter()
                self.client_stub.ReceiveResponse(response)
                
                self.grpc_overhead = time.perf_counter() - start
                self.chain.update_grpc_overhead(self.grpc_overhead)

            except grpc.RpcError as e:
                logger.error(f"Failed to send result to client at {response_address}: {e}")
        
            return 

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

        request.response_address = response_address

        try:
            logger.info(f"Forwarding RunLayers request to successor from ({response_address})...")

            start = time.perf_counter()
            self.successor_stub.RunLayers(request, timeout=5)

            self.grpc_overhead = time.perf_counter() - start
            self.chain.update_grpc_overhead(self.grpc_overhead)
        except grpc.RpcError as e:
            if (
                e.code() == grpc.StatusCode.UNAVAILABLE
                or e.code() == grpc.StatusCode.DEADLINE_EXCEEDED
            ):
                logger.warning(f"Successor failure detected during INFERENCE.")
                threading.Thread(target=self.repair_chain, daemon=True).start()
                self._connect_to_client(response_address)
                self.client_stub.ReceiveResponse(
                    nodeservice_pb2.InferenceResponse(error_message=f"Node with address: {self.successor_stub_addr} has failed. The chain is being repaired...")
                )

        return

    def reallocate_layers(
        self,
        total_system_rate: float,
        start_layer_index: int,
        load_max: bool,
        predecessor_info: Optional[Dict[str, Any]],
    ) -> nodeservice_pb2.ReallocateResponse:
        """
        Executes the AR-MDI logic
        """
        logger.info(
            f" REALLOCATION TRIGGERED:\nTotal Rate: {total_system_rate} | My Rate: {self.processing_rate} | Start Index: {start_layer_index}"
        )
        # succ_info = self.chain.get_successor_info()

        # if succ_info:
        #     successor_proc_rate = succ_info.get("processing_rate")
        #     succ_layers = succ_info.get("layers")
        #     succ_output_layer_loaded = succ_info.get("output_layer_loaded", False)

        #     # If this node's processing rate is much smaller then its successor's
        #     # the successor should take this node's layers
        #     if successor_proc_rate > self.processing_rate * REALLOC_TAKEOVER_MULT_THRESHOLD:
        #         # Only if the successor can load the layers
        #         if can_load(
        #             self.config,
        #             succ_info.get("available_memory"),
        #             succ_info.get("available_vram"),
        #             layers=self.llm.layers_loaded
        #         ):
        #             if succ_layers:
        #                 new_layers = (self.llm.layers_loaded[0], succ_layers[1])
        #             else:
        #                 new_layers = self.llm.layers_loaded

        #             self._unload_llm()

        #             self.chain.repair(
        #                 new_layers,
        #                 replacement_info=succ_info,
        #                 replacee_info=self.chain.get_self_info(),
        #                 replacement_load_output_layer=succ_output_layer_loaded,
        #                 make_replacee_backup=True,
        #                 replacee_was_head=self.chain.is_head(),
        #                 replacee_pred_info=predecessor_info,
        #             )

        #             total_system_rate -= self.processing_rate

        #             # Forward reallocation request
        #             serialized_pred_info = json.dumps(predecessor_info).encode("utf-8")

        #             realloc_request = nodeservice_pb2.ReallocateRequest(
        #                 total_rate=total_system_rate,
        #                 start_layer_index=start_layer_index,
        #                 predecessor_info=serialized_pred_info,
        #             )
        #             try:
        #                 self.successor_stub.LoadLayers(nodeservice_pb2.Empty())

        #                 response = self.successor_stub.Reallocate(realloc_request)
        #                 return response
        #             except grpc.RpcError as e:
        #                 logger.error(f"Failed to propagate Reallocation to successor: {e}")
        #                 return nodeservice_pb2.ReallocateResponse(success=False)

        # ---- Reallocation ----
        num_total_layers = self.config.get("num_hidden_layers")

        # tail_output_temporal_tle = self.chain.get_tail_output_temporal_tle()
        # logger.info(f"Tail Output Temporal TLE: {tail_output_temporal_tle}")
        logger.info(f"Output Layer Memory TLE: {self.output_layer_memory_tle}")
        
        num_total_tle = num_total_layers + self.output_layer_memory_tle

        # Calculate share
        if total_system_rate > 0:
            # Epic equation
            ideal_tle_count = num_total_tle * (self.processing_rate / total_system_rate)
        else:
            ideal_tle_count = 0

        logger.info(f"Ideal TLE Count: {ideal_tle_count}")


        # Calculate the max number of tle this node can load
        current_num_layers = self.model.num_layers
        if self.model.output_layer_loaded:
            current_num_layers += self.output_layer_memory_tle

        theoretical_mem_in_use = current_num_layers * self.layer_mem_size_mb + RESERVED_MEM_MB
        usage = self.vram_usage_mb if self.vram_usage_mb > 0 else self.memory_usage_mb

        mem_in_use = theoretical_mem_in_use if theoretical_mem_in_use > usage else usage

        available_memory = None
        available_vram = None
        if self.available_vram_mb > 0:
            available_vram = usage + self.available_vram_mb - mem_in_use
        else:
            available_memory = usage + self.available_memory_mb - mem_in_use

        max_extra_num_layers = self._mem_to_num_layers(
            available_vram=available_vram,
            available_memory=available_memory,
            round_it=False,
        )

        max_num_layers = current_num_layers + max_extra_num_layers
        logger.info(f"Max Num Transformer Layers (Memory Limit): {max_num_layers}")

        
        target_tle_count = min(ideal_tle_count, max_num_layers)
        logger.info(f"Target TLE Count: {target_tle_count}")

        
        tail_cant_load_remaining = False
        if self.chain.is_tail():
            remaining_num_trans_layers = num_total_layers - start_layer_index if start_layer_index > 0 else 0
            target_transformer_layer_count = target_tle_count - self.output_layer_memory_tle

            if target_transformer_layer_count < remaining_num_trans_layers:
                # adjust rate, restart realloc
                tail_cant_load_remaining = True
        
        if tail_cant_load_remaining:
            logger.warning("Tail can't load the Output Layer after Loading the remaining Layers!")
            
        
        # If the ideal number of parameters dont fit in memory, then its like this node had a lower processing rate
        # A node shouldnt restart the reallocation if its both the head and tail
        memory_limit_exceeded = round(ideal_tle_count, 2) > round(max_num_layers, 2)
        if memory_limit_exceeded:
            logger.warning(f"Memory Limit Hit!")
        is_isolated_node = not (self.chain.is_head() and self.chain.is_tail())

        if ((memory_limit_exceeded and is_isolated_node) or tail_cant_load_remaining) and not load_max:
            remaining_rate = total_system_rate - self.processing_rate
            effective_num_layers = max_num_layers if memory_limit_exceeded else remaining_num_trans_layers + self.output_layer_memory_tle

            effective_rate = effective_num_layers * remaining_rate / ((num_total_tle) - effective_num_layers)
        
            logger.warning(f"Effective Num Layers: {effective_num_layers} | Effective Processing Rate: {effective_rate}. Requesting reallocation restart...")
            self.processing_rate = effective_rate
            self.chain.update_processing_rate(effective_rate)
            return nodeservice_pb2.ReallocateResponse(
                success=False,
                requires_restart=True,
                bottleneck_node=self.grpc_addr
            )

        # Start with transformer layers
        target_layer_count = int(round(target_tle_count))
        
        end_layer_index = start_layer_index + target_layer_count - 1
        if end_layer_index >= num_total_layers:
            end_layer_index = num_total_layers - 1
            target_layer_count = end_layer_index - start_layer_index + 1 
        

        # Check if the lm_head should be loaded
        load_output_layer = False
        if self.chain.is_tail() and max_num_layers >= (target_layer_count + self.output_layer_memory_tle) and end_layer_index == num_total_layers - 1:
            load_output_layer = True
        
        new_layers = None
        if start_layer_index < num_total_layers:
            new_layers = (start_layer_index, end_layer_index)
        logger.info(f"Target Layer Count: {target_layer_count} | New Layers: {new_layers} | Load Output Layer: {load_output_layer}")

        # Load Layers
        self._reload_llm(layers=new_layers, load_output_layer=load_output_layer)
        self.chain.update_layers(new_layers)

        next_start_index = end_layer_index + 1

        # Propagate to successor
        if not self.chain.is_tail() and self.successor_stub:
            serialized_pred_info = json.dumps(self.chain.get_self_info()).encode("utf-8")

            request = nodeservice_pb2.ReallocateRequest(
                total_rate=total_system_rate,
                start_layer_index=next_start_index,
                load_max=load_max,
                predecessor_info=serialized_pred_info,
            )
            try:
                response = self.successor_stub.Reallocate(request)
                return response
            except grpc.RpcError as e:
                logger.error(f"Failed to propagate Reallocation to successor: {e}")
                return nodeservice_pb2.ReallocateResponse(success=False)
        else:
            return nodeservice_pb2.ReallocateResponse(
                success=True,
                requires_restart=False
            )

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

            dead_successor_info = self.chain.get_successor_info()

            if not dead_successor_info:
                logger.error("Could not retrieve successor data from DHT! Chain is broken.")
                return

            dead_succ_was_tail = self.chain.node_is_tail(dead_successor_info.get("id"))
            dead_succ_output_loaded = dead_successor_info.get("output_layer_loaded", False)

            self.successor_stub = None

            self.chain.dht.store(ALL_LAYERS_KEY, False, EXPIRATION_S)
            self.chain.become_chain_leader()
            self.chain.update_chain_status(ChainStatus.REPAIRING)

            # Repair
            orphaned_layers = dead_successor_info.get("layers")
            logger.info(f"Attempting to repair chain. Orphaned layers: {orphaned_layers}...")

            # Check if this node has enough memory to load the orphaned layers
            if self._can_load(layers=orphaned_layers, output_layer=dead_succ_output_loaded):
                if orphaned_layers:
                    layers_to_load = (self.llm.layers_loaded[0], orphaned_layers[1])
                elif dead_succ_output_loaded:
                    layers_to_load = self.llm.layers_loaded
                
                logger.info(f"Taking over layers {orphaned_layers}. New range: {layers_to_load}. Load Output Layer: {dead_succ_output_loaded}")

                self._reload_llm(layers_to_load, load_output_layer=dead_succ_output_loaded)
                
                self_info = self.chain.get_self_info()
                self.chain.repair(
                    layers_to_load,
                    replacement_info=self_info,
                    replacee_info=dead_successor_info,
                    replacee_was_tail=dead_succ_was_tail,
                )

                self._connect_to_successor()

                if self.chain.get_all_layers_loaded():
                    self.chain.update_chain_status(ChainStatus.READY)
                else:
                    self.chain.update_chain_status(ChainStatus.UNREADY)
                return

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
                            logger.info(
                                f"Backup Node {backup_info.get('id')[:DIGITS_SHOW]} found with Available Memory: {avail_mem} MB and Available VRAM: {avail_vram} MB"
                            )

                            if can_load(
                                config=self.config,
                                current_layers=backup_info.get("layers"),
                                output_layer_curr_loaded=backup_info.get("output_layer_loaded"),
                                avail_mem=avail_mem,
                                mem_usage=backup_info.get("memory_usage"),
                                avail_vram=avail_vram,
                                vram_usage=backup_info.get("vram_usage"),
                                layers=orphaned_layers,
                                output_layer=dead_succ_output_loaded
                            ):
                                self_info = self.chain.get_self_info()
                                self.chain.repair(
                                    orphaned_layers,
                                    replacement_info=backup_info,
                                    replacement_load_output_layer=dead_succ_output_loaded,
                                    replacee_info=dead_successor_info,
                                    replacee_was_tail=dead_succ_was_tail,
                                    replacee_pred_info=self_info,
                                )

                                try:
                                    self._connect_to_successor()
                                    self.successor_stub.LoadLayers(nodeservice_pb2.Empty())

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
                                logger.info(
                                    f"Backup Node {backup_id} cannot load the orphaned layers."
                                )
                else:
                    logger.info("No backup nodes found. Setting this node as the tail...")

                # If no backup node is found or the backup cant load the orphaned layers, set this node as the TAIL
                self.chain.update_chain_tail(self.chain.node_id)
                self.chain.update_successor(new_successor_data=None)
                # self.chain.update_all_layers_loaded()
                

                # Also make every node that succeeded the dead node a backup
                succ_data = dead_successor_info.get("successor")
                if succ_data:
                    logger.info("Making every node that succeeded the dead node a backup...")
                    current_node_id = succ_data.get("id")
                    while current_node_id:
                        current_node_info = self.chain.get_server_info(current_node_id)
                        current_node_succ_data = current_node_info.get("successor")
                        if current_node_succ_data:
                            next_node_id = current_node_succ_data.get("id")
                        else:
                            next_node_id = None
                        
                        self.chain.make_node_backup(current_node_id)
                        
                        try:
                            channel = grpc.insecure_channel(current_node_info.get("address"))
                            curr_node_stub = nodeservice_pb2_grpc.NodeServiceStub(channel)
                            curr_node_stub.UnloadLayers(nodeservice_pb2.Empty())
                        except grpc.RpcError as e:
                            logger.error(
                                f"A gRPC error occurred while connecting to {current_node_info.get('address')}: {e.code().name}"
                            )
                        
                        current_node_id = next_node_id
                
                # Make the new Backup Nodes Join the Chain 
                logger.info(f"Attempting to rebuild the chain with backup nodes...")
                backup_nodes = self.chain.get_backup_nodes()
                if backup_nodes:
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
                
                if self.chain.get_all_layers_loaded():
                    self.chain.update_chain_status(ChainStatus.READY)
                else:
                    self.chain.update_chain_status(ChainStatus.UNREADY)
                

    def opportunistic_takeover(
        self, weak_node_info: Dict[str, Any], predecessor_info: Optional[Dict[str, Any]]
    ):
        """
        After a backup node finds a weak node in the chain to replace.
        1. It loads the layers of the weak node.
        2. The backup node takes its place in the active chain by making the neccesary changes to the dht.
        3. It sends a grpc request to its predecessor to update its successor_stub attribute.
        3. It sends a grpc request to the weak node, to unload its layers.
        """

        layers_to_takeover = weak_node_info.get("layers")
        weak_node_output_layer_loaded = weak_node_info.get("output_layer_loaded", False)

        weak_node_was_tail = self.chain.node_is_tail(weak_node_info.get("id"))
        weak_node_was_head = self.chain.node_is_head(weak_node_info.get("id"))
        logger.info(
            f"Node: {self.chain.node_id[:DIGITS_SHOW]} attempting to takeover layers {layers_to_takeover}."
        )

        if self._can_load(layers=layers_to_takeover, output_layer=weak_node_output_layer_loaded):
            self.chain.dht.store(ALL_LAYERS_KEY, False, EXPIRATION_S)

            self.chain.become_chain_leader()
            self.chain.update_chain_status(ChainStatus.TAKEOVER)

            if self.llm is not None and self.llm.layers_loaded is not None:
                new_layers = (self.llm.layers_loaded[0], layers_to_takeover[1]) if layers_to_takeover else self.llm.layers_loaded
            else:
                new_layers = layers_to_takeover
            
            self._reload_llm(new_layers, load_output_layer=weak_node_output_layer_loaded)
            
            self.chain.repair(
                new_layers,
                replacement_info=self.chain.get_self_info(),
                replacee_info=weak_node_info,
                make_replacee_backup=True,
                replacee_was_head=weak_node_was_head,
                replacee_was_tail=weak_node_was_tail,
                replacee_pred_info=predecessor_info,
            )

            # Update the predecessor's successor_stub to point to this node
            if predecessor_info:
                try:
                    channel = grpc.insecure_channel(predecessor_info.get("address"))
                    predecessor_stub = nodeservice_pb2_grpc.NodeServiceStub(channel)
                    predecessor_stub.UpdateSuccessorStub(nodeservice_pb2.Empty())
                except grpc.RpcError as e:
                    logger.error(
                        f"A gRPC error occurred while connecting to {predecessor_info.get('address')}: {e.code().name}"
                    )

            # Send GRPC request to the weak node to unload its layers
            try:
                channel = grpc.insecure_channel(weak_node_info.get("address"))
                weak_node_stub = nodeservice_pb2_grpc.NodeServiceStub(channel)
                weak_node_stub.UnloadLayers(nodeservice_pb2.Empty())
            except grpc.RpcError as e:
                logger.error(
                    f"A gRPC error occurred while connecting to {weak_node_info.get('address')}: {e.code().name}"
                )

            self._connect_to_successor()

        else:
            logger.info(
                f"Node: {self.chain.node_id} can not load layers {layers_to_takeover} of the weak node."
            )

        # Update chain status
        if self.chain.get_all_layers_loaded():
            self.chain.update_chain_status(ChainStatus.READY)
        else:
            self.chain.update_chain_status(ChainStatus.UNREADY)

    def _connect_to_successor(self):
        """Establishes a gRPC connection to the successor node."""
        # Check if the successor stub exists and its address corresponds to the successor address stored on the dht.
        # After an opportunistic takeover they would be different.
        if (
            self.successor_stub is not None
            and self.successor_stub_addr == self.chain.get_successor_address()
        ):
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
            channel = create_grpc_channel(successor_addr)
            grpc.channel_ready_future(channel).result(timeout=10)
            self.successor_stub = nodeservice_pb2_grpc.NodeServiceStub(channel)
            self.successor_stub_addr = successor_addr
            logger.info(f"Connection to successor: {successor_addr} established.")
        
        except grpc.FutureTimeoutError:
            logger.error(f"Connection to {successor_addr} timed out.")
            self.successor_stub = None

            raise CustomRpcError(
                code=grpc.StatusCode.DEADLINE_EXCEEDED,
                details="Connection to successor timed."
            )

        except grpc.RpcError as e:
            logger.error(
                f"A gRPC error occurred while connecting to {successor_addr}: {e.code().name}"
            )
            self.successor_stub = None
    
    def _connect_to_client(self, client_addr: str):
        """Establishes a gRPC connection to a client node."""       
        if not self.client_stub or self.client_stub_addr != client_addr:
            channel = create_grpc_channel(client_addr)
            self.client_stub_addr = client_addr
            self.client_stub = nodeservice_pb2_grpc.ClientServiceStub(channel)


UDP_PORT = 9999

# To not show netlinkrib errors on moto
os.environ["GOLOG_LOG_LEVEL"] = "fatal"


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

    my_ip = os.getenv("IP")
    if not my_ip:
        my_ip = get_ip_address()
    
    grpc_port = os.getenv("GRPC_PORT")
    grpc_port = grpc_port if grpc_port else get_free_port()
    grpc_addr = f"{my_ip}:{grpc_port}"

    # host_maddrs = os.getenv("HOST_MADDRS")
    host_maddrs = f"/ip4/{my_ip}/tcp/0"

    bootstrap_node_addr_str = os.getenv("BOOTSTRAP_NODE_ADDR")

    if not bootstrap_node_addr_str:
        bootstrap_node_addr_str = discover_bootstrap_node_address()

    if not bootstrap_node_addr_str:
        logger.warning("Failed to find bootstrap node address.")

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
    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=GPRC_MAX_WORKERS),
        options=[
            ("grpc.max_send_message_length", GRPC_MAX_MSG_SIZE),
            ("grpc.max_receive_message_length", GRPC_MAX_MSG_SIZE),
        ],
    )
    nodeservice_pb2_grpc.add_NodeServiceServicer_to_server(NodeServicer(server_node), grpc_server)
    grpc_server.add_insecure_port(grpc_addr)

    grpc_server.start()
    logger.info("Server is ready to accept grpc connections.")

    # Initialize the successor stub
    server_node._connect_to_successor()

    def _dht_heartbeat_task(server_node: Server):
        """Background task to keep DHT keys alive."""
        while True:
            time.sleep(HEARTBEAT_INTERVAL_S)
            server_node._update_memory_usage()
            if server_node.llm is not None and hasattr(server_node.llm, "output_layer_loaded"):
                server_node.chain.update_layers_loaded(server_node.model.num_layers>0)
                server_node.chain.update_output_layer_loaded(server_node.llm.output_layer_loaded)
            server_node.chain.republish_keys()

    def _chain_health_monitor_task(server_node: Server):
        """Backgroud task to check on the node's successor status"""
        while True:
            time.sleep(HEARTBEAT_INTERVAL_S)
            if server_node.chain.is_backup():
                # --- Backup Opportunistic Takeover ---
                weak_node_info, predecessor_info = server_node.chain.evaluate_takeover_eligibility()

                if weak_node_info and server_node.chain.get_chain_status() == ChainStatus.READY:
                    server_node.opportunistic_takeover(weak_node_info, predecessor_info)

            elif server_node.chain.is_tail():
                continue

            # Perform the health check on the successor
            if server_node.chain.get_chain_status() in (ChainStatus.READY, ChainStatus.UNREADY, ChainStatus.NONE):
                try:
                    server_node._connect_to_successor()
                    if server_node.successor_stub is not None:
                        server_node.successor_stub.Check(nodeservice_pb2.Empty(), timeout=5)
                        logger.info(f"Successor is ALIVE.")
                except grpc.RpcError as e:
                    if (
                        e.code() == grpc.StatusCode.UNAVAILABLE
                        or e.code() == grpc.StatusCode.DEADLINE_EXCEEDED
                    ):
                        logger.warning(f"Successor failure detected during HEALTH CHECK.")
                        server_node.repair_chain()
                    else:
                        logger.warning(
                            f"A gRPC error occurred during health check: {e.code().name}"
                        )
                        server_node.successor_stub = None

            # --- Active Node Successor Opportunistic Takeover ---
            if server_node.chain.get_chain_status() == ChainStatus.READY:
                succ_info = server_node.chain.get_successor_info()
                if not succ_info:
                    continue

                successor_proc_rate = succ_info.get("processing_rate")
                if server_node.processing_rate > successor_proc_rate * ACTIVE_NODE_TAKEOVER_MULT_THRESHOLD:
                    server_node.opportunistic_takeover(succ_info, predecessor_info=None)

    def _udp_discovery_server():
        """Background task that listens for bootstrap discovery requests and responds with the servers grpc address"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", UDP_PORT))

        response_b = grpc_addr.encode()

        while True:
            data, addr = sock.recvfrom(1024)
            if data == b"DISCOVER_BOOTSTRAP":
                sock.sendto(response_b, addr)

    chain_monitor_thread = threading.Thread(
        target=_chain_health_monitor_task, args=(server_node,), daemon=True
    )
    chain_monitor_thread.start()

    dht_heartbeat_thread = threading.Thread(
        target=_dht_heartbeat_task, args=(server_node,), daemon=True
    )
    dht_heartbeat_thread.start()

    udp_discovery_thread = threading.Thread(target=_udp_discovery_server, daemon=True)
    udp_discovery_thread.start()

    # Shutdown handler
    def _handle_shutdown(signum, frame):
        # server.stop(grace=2)

        # Its not realistic to expect a grace period in a real distributed system node failure
        logger.info("Shutting down gRPC server...")
        grpc_server.stop(grace=None)

        # server_node.dht.shutdown()

    signal.signal(signal.SIGINT, _handle_shutdown)  # Ctrl+C
    signal.signal(signal.SIGTERM, _handle_shutdown)  # docker stop

    # Profile the node to get processing rate measurements
    profile_node_str = os.getenv("PROFILE")
    profile_node = int(profile_node_str) if profile_node_str is not None else 1
    if profile_node:
        server_node._profile_node(dummy_seq_length=50, profiling_runs=50, profiling_duration_s=PROFILING_DURATION)

    # Load the server's assigned layers
    # if not server_node.chain.is_backup():
    server_node._load_llm()

    grpc_server.wait_for_termination()


if __name__ == "__main__":
    serve()
