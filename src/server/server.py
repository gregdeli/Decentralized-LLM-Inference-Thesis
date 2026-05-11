"""Main server application logic"""

import json
import logging
import sys
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
    update_config_layer_param_count,
    update_config_final_output_param_count,
    update_config_total_param_count,
    can_load
)
from core.remote.utils import (
    get_bootstrap_peer_address,
    get_ip_address,
    discover_bootstrap_node_address,
    create_grpc_channel,
    get_free_port
)
from core.remote import nodeservice_pb2, nodeservice_pb2_grpc
from core.remote.serialization import *
from core.p2p.dht_manager import DHTManager
from core.p2p.chain_manager import (
    ChainManager,
    ChainStatus,
    HEARTBEAT_INTERVAL_S,
    ALL_LAYERS_KEY,
    NUM_CLIENTS_KEY,
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

        self.model_path = model_path
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        if torch.cuda.is_available():
            try:
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

        # Determine Model Parameter Load Capacity
        max_num_params = self._mem_to_num_params()

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
        }

        self.chain.join_chain(
            server_info,
            max_num_layers=num_layers,
            max_num_params=max_num_params,
            num_total_layers=self.config.get("num_hidden_layers"),
            num_total_params=self.config.get("num_total_params"),
            transformer_layer_params=self.config.get("transformer_layer_params"),
            final_output_params=self.config.get("final_output_params")

        )

        # Initialize State
        self.llm = None
        self.model = None
        self.successor_stub = None
        self.successor_stub_addr = None  # Address that corresponds to the successor_stub. To check if same with dht succ. addr.
        self.num_local_params = 0

        self.inference_delay = 0.0
        self.processing_rate = 0
        self.grpc_overhead = 0.0
        # self.succ_network_latency = 0.0

        # Client Stub
        self.client_stub = None
        self.client_stub_addr = None
    def _load_llm(
        self,
        time_it: bool = False,
    ):
        """Loads the layers assigned by the ChainManager."""
        layers_to_load = self.chain.get_layers()
        load_output_layer = self.chain.get_load_output_layer()

        if self.chain.is_backup():
            # Load only a single layer for fast but still accurate profiling
            layers_to_load = (0, 0)

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
        if not self.chain.is_backup():
            self.chain.update_layers_loaded(self.model.num_layers>0)
            if self.llm.output_layer_loaded:
                self.chain.update_load_output_layer(False)
            self.chain.update_output_layer_loaded(self.llm.output_layer_loaded)
            self.chain.update_all_layers_loaded()

    def _unload_llm(self):
        """Helper to unload the model with explicit GC"""
        self.model = None
        self.llm = None
        self.num_local_params = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._update_memory_usage()

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
        self.chain.update_layers_loaded(self.model.num_layers>0)
        self.chain.update_output_layer_loaded(self.llm.output_layer_loaded)

    def _reload_config(self):
        with open(self.model_path / "config.json", "r") as f:
            self.config = json.load(f)

    def _mem_to_num_params(self) -> int:
        """
        Calculates how many model parameters fit in the available Memory/VRAM
        """
        self._update_memory_usage(update_on_dht=False)

        # Get the number of total model parameters from the model config 
        # or add it if it doesnt exist
        total_params = self.config.get("num_total_params")
        if total_params is None:
            layer_params = self.config.get("transformer_layer_params")
            if layer_params is None:
                layer_params = calculate_transformer_params(self.model_path)
                update_config_layer_param_count(self.model_path, layer_params)
                self._reload_config()
            
            final_output_params = self.config.get("final_output_params")
            if final_output_params is None:
                final_output_params = calculate_final_output_params(self.model_path)
                update_config_final_output_param_count(self.model_path, final_output_params)
                self._reload_config()
            
            total_params = self.config.get("num_hidden_layers") * layer_params + final_output_params
            update_config_total_param_count(self.model_path, total_params)
            self._reload_config()

        if self.device == "cuda":
            available = self.available_vram_mb - RESERVED_MEM_MB
        else:
            available = self.available_memory_mb - RESERVED_MEM_MB

        if available <= 0:
            return 0
        
        param_dtype = get_dtype_from_config(self.config)
        bytes_per_param = param_dtype.itemsize

        max_num_params = available / (bytes_per_param / (1024 * 1024)) 
        return min(max_num_params, total_params)


    def _mem_to_num_layers(self) -> int:
        """
        Calculates how many transformer layers fit in the available Memory/VRAM taking into account the KV Cache 
        """
        self._update_memory_usage(update_on_dht=False)

        if "head_dim" not in self.config:
            self.config["head_dim"] = self.config["hidden_size"] // self.config["num_attention_heads"]

        layer_params = self.config.get("transformer_layer_params")

        param_dtype = get_dtype_from_config(self.config)
        bytes_per_param = param_dtype.itemsize
        
        layer_param_mem_size_mb = (layer_params * bytes_per_param) / (1024 * 1024)
        layer_kv_cache_mem_size_mb = (2 * self.config.get("num_key_value_heads") * GLOBAL_MAX_SEQ_LEN * self.config.get("head_dim") * bytes_per_param) / (1024 * 1024)
        layer_mem_size_mb = layer_param_mem_size_mb + layer_kv_cache_mem_size_mb

        if self.device == "cuda":
            available = self.available_vram_mb - RESERVED_MEM_MB
        else:
            available = self.available_memory_mb - RESERVED_MEM_MB

        if available <= 0:
            return 0

        max_num_layers = int(available // layer_mem_size_mb)
        return min(max_num_layers, self.config.get("num_hidden_layers"))

    def _can_load(
        self,
        num_layers: Optional[int] = None,
        layers: Optional[Tuple[int, int]] = None,
        output_layer: Optional[bool] = False,
    ) -> bool:
        """Checks if this node can load a certain number of layers or a range of layers"""
        num_layers = num_layers if num_layers else (layers[1] - layers[0] + 1) if layers else 0
        max_num_params = self._mem_to_num_params()

        if num_layers > 0:
            max_num_layers = self._mem_to_num_layers()

            if max_num_layers < num_layers:
                return False
            
            if output_layer:
                max_num_params -= num_layers * self.config.get("transformer_layer_params")
                return max_num_params >= self.config.get("final_output_params")
            
            return True
        
        return max_num_params >= self.config.get("final_output_params")

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
            num_clients = self.chain.dht.get(NUM_CLIENTS_KEY)
            max_seq_length = GLOBAL_MAX_SEQ_LEN // num_clients if num_clients else GLOBAL_MAX_SEQ_LEN

            self.model.add_client_cache(
                client_id, 
                batch_size=1,
                max_seq_length=max_seq_length,
                device=self.device,
                dtype=self.llm.dtype
            )

            self._update_memory_usage(update_on_dht=True)

        # # Initialize the kv cache if necessary
        # if not self.llm.kv_cache_initialized:
        #     self.model.set_kv_cache(
        #         batch_size=1,
        #         # max_seq_length=max_returned_tokens,
        #         max_seq_length=MAX_SEQUENCE_LENGTH,
        #         device=self.device,
        #         dtype=self.llm.dtype,
        #     )
        #     self.llm.kv_cache_initialized = True
        #     # self.llm.prev_max_seq_length = max_returned_tokens
        #     cache_updated = True

        # Dynamically grow the kv cache size if necessary
        # elif max_returned_tokens > self.llm.prev_max_seq_length:
        #     tmp_device = self.model.mask_cache.device
        #     self.model.clear_kv_cache()
        #     self.model.set_kv_cache(
        #         batch_size=1,
        #         max_seq_length=max_returned_tokens,
        #         device=tmp_device,
        #         dtype=self.llm.dtype,
        #     )
        #     self.llm.prev_max_seq_length = max_returned_tokens
        #     cache_updated = True

        # if cache_updated:
        #     self._update_memory_usage(update_on_dht=True)

    def _calculate_processing_rate(self, delay: float):
        """Calculates and updates the moving average of layers processed per second."""
        if delay > 0 and self.num_local_params > 0:
            current_rate = int(self.num_local_params / delay)

            # Moving average to avoid jitter
            if self.processing_rate == 0:
                self.processing_rate = current_rate
            else:
                self.processing_rate = int((0.7 * self.processing_rate) + (0.3 * current_rate))

    @torch.no_grad()
    def _profile_node(
        self, 
        dummy_seq_length: int = 30, 
        profiling_runs: int = 100, 
        profiling_duration_s: int = 3
    ):
        """Measures a backup node's processing rate by doing a fake generation on a dummy input."""
        # LLM has already been loaded for profiling with the number of layers the backup node can hold

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

            self.inference_delay = time.perf_counter() - start

            self._calculate_processing_rate(self.inference_delay)

            logger.info(
                f"Profiling: Layers {self.llm.layers_loaded}: "
                f"Delay: {self.inference_delay:.4f}s "
                f"Params/sec: {self.processing_rate} "
            )

            dummy_input = torch.randn(1, 1, hidden_size, device=self.device, dtype=self.llm.dtype)
            current_pos = starting_dummy_seq_len + (i + 1)
            dummy_input_pos = torch.tensor([current_pos], device=self.device)
            dummy_seq_length = 1

            profiling_time = time.perf_counter() - prof_start_time
            if profiling_time >= profiling_duration_s:
                break

        self.chain.update_processing_rate(self.processing_rate)
        self.model.remove_client_cache("profiling")
        self._update_memory_usage()

        if self.chain.is_backup():
            logger.info(f"Backup Node Profiling Complete")
            self._unload_llm()

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
        self.chain.update_chain_status(ChainStatus.RUNNING)

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
        if not self.chain.get_all_layers_loaded():
            self.chain.update_chain_status(ChainStatus.UNREADY)
            self._connect_to_client(response_address)
            self.client_stub.ReceiveResponse(
                nodeservice_pb2.InferenceResponse(error_message="Inference requested without all the layers being loaded!")
            )
            return

        # If TAIL Node -> Send response to Client
        if self.chain.is_tail() and self.chain.get_output_layer_loaded():
            logits = h
            next_token = self.llm.sample_logits(logits)

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
        predecessor_info: Optional[Dict[str, Any]],
    ):
        """
        Executes the AR-MDI logic
        """
        logger.info(
            f" REALLOCATION TRIGGERED:\nTotal Rate: {total_system_rate} | My Rate: {self.processing_rate} | Start Index: {start_layer_index}"
        )
        REALLOC_TAKEOVER_MULT_THRESHOLD = 2.0
        succ_info = self.chain.get_successor_info()

        if succ_info:
            successor_proc_rate = succ_info.get("processing_rate")
            succ_layers = succ_info.get("layers")
            succ_output_layer_loaded = succ_info.get("output_layer_loaded", False)

            # If this node's processing rate is much smaller then its successor's
            # the successor should take this node's layers
            if successor_proc_rate > self.processing_rate * REALLOC_TAKEOVER_MULT_THRESHOLD:
                new_layers = (self.llm.layers_loaded[0], succ_layers[1])

                self._unload_llm()

                self.chain.repair(
                    new_layers,
                    replacement_info=succ_info,
                    replacee_info=self.chain.get_self_info(),
                    replacement_load_output_layer=succ_output_layer_loaded,
                    make_replacee_backup=True,
                    replacee_was_head=self.chain.is_head(),
                    replacee_pred_info=predecessor_info,
                )

                total_system_rate -= self.processing_rate

                # Forward reallocation request
                serialized_pred_info = json.dumps(predecessor_info).encode("utf-8")

                request = nodeservice_pb2.ReallocateRequest(
                    total_rate=total_system_rate,
                    start_layer_index=start_layer_index,
                    predecessor_info=serialized_pred_info,
                )
                try:
                    self.successor_stub.Reallocate(request)
                    return
                except grpc.RpcError as e:
                    logger.error(f"Failed to propagate Reallocation to successor: {e}")

            # If this node's processing rate is much greater than the proceccing rate of its successor
            # it should take its layers and make it a backup
            elif self.processing_rate > successor_proc_rate * REALLOC_TAKEOVER_MULT_THRESHOLD:
                logger.info(
                    f"Attempting to takeover layers: {succ_layers} from successor {succ_info.get('id')[:DIGITS_SHOW]}..."
                )

                if self._can_load(layers=succ_layers, output_layer=succ_output_layer_loaded):
                    layers_to_load = (self.llm.layers_loaded[0], succ_layers[1])
                    logger.info(f"Taking over layers {succ_layers}. New range: {layers_to_load}. Load Output Layer: {succ_output_layer_loaded}")

                    self._reload_llm(layers_to_load, load_output_layer=succ_output_layer_loaded)

                    self.chain.repair(
                        new_layers=layers_to_load,
                        replacement_info=self.chain.get_self_info(),
                        replacee_info=succ_info,
                        replacement_load_output_layer=succ_output_layer_loaded,
                        make_replacee_backup=True,
                        replacee_was_tail=self.chain.node_is_tail(succ_info.get("id")),
                    )

                    # Send GRPC request to the weak node to unload its layers
                    self.successor_stub.UnloadLayers(nodeservice_pb2.Empty())

                    self._connect_to_successor()

                    total_system_rate -= successor_proc_rate

                else:
                    logger.info("Cannot load layers. Continuing with the layer reallocation...")

        # ---- Reallocation ----
        num_total_params = self.config.get("num_total_params")
        num_total_layers = self.config.get("num_hidden_layers")

        # Calculate share
        if total_system_rate > 0:
            # Epic equation
            ideal_param_count = num_total_params * (self.processing_rate / total_system_rate)
        else:
            ideal_param_count = 0

        logger.info(f"Ideal Parameter Count: {ideal_param_count}")

        current_num_params = self.num_local_params
        max_extra_num_params = self._mem_to_num_params()
        max_num_params = current_num_params + max_extra_num_params
        logger.info(f"Max Num Parameters: {max_num_params}")

        current_num_layers = self.model.num_layers
        max_extra_num_layers = self._mem_to_num_layers()
        max_num_layers = current_num_layers + max_extra_num_layers
        logger.info(f"Max Num Layers: {max_num_layers}")

        # Don't exceed the number of layers that can be loaded
        target_param_count = min(ideal_param_count, max_num_params)
        
        logger.info(f"Target Param Count: {target_param_count}")

        transformer_layer_params = self.config.get("transformer_layer_params")
        final_output_params = self.config.get("final_output_params")

        # Start with transformer layers
        target_layer_count = min(target_param_count / transformer_layer_params, max_num_layers)
        
        end_layer_index = start_layer_index + int(round(target_layer_count)) - 1
        if end_layer_index >= num_total_layers:
            end_layer_index = num_total_layers - 1
            target_layer_count = end_layer_index - start_layer_index + 1 

        # Check if the lm_head should be loaded
        max_num_params -= target_layer_count * transformer_layer_params
        target_param_count -= target_layer_count * transformer_layer_params

        load_output_layer = False
        # if remaining_param_count >= final_output_params:
        if (self.chain.is_tail() and max_num_params >= final_output_params) or target_param_count>=final_output_params:
            if end_layer_index == num_total_layers - 1:
                load_output_layer = True
        
        new_layers = (start_layer_index, end_layer_index)
        logger.info(f"Target Layer Count: {target_layer_count} | New Layers: {new_layers} | Load Output Layer: {load_output_layer}")

        # Load Layers
        if new_layers != self.llm.layers_loaded or load_output_layer != self.llm.output_layer_loaded:
            self._reload_llm(new_layers, load_output_layer)
            self.chain.update_layers(new_layers)
        else:
            logger.info("Layer assignment unchanged.")

        next_start_index = end_layer_index + 1

        # Propagate to successor
        if not self.chain.is_tail() and self.successor_stub:
            serialized_pred_info = json.dumps(self.chain.get_self_info()).encode("utf-8")

            request = nodeservice_pb2.ReallocateRequest(
                total_rate=total_system_rate,
                start_layer_index=next_start_index,
                predecessor_info=serialized_pred_info,
            )
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

            dead_successor_info = self.chain.get_successor_info()

            if not dead_successor_info:
                logger.error("Could not retrieve successor data from DHT! Chain is broken.")
                return

            dead_succ_was_tail = self.chain.node_is_tail(dead_successor_info.get("id"))
            dead_succ_output_loaded = dead_successor_info.get("output_layer_loaded", False)

            self.successor_stub = None
            self.chain.dht.store(ALL_LAYERS_KEY, False, EXPIRATION_S)
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
                    replacement_load_output_layer=dead_succ_output_loaded,
                    replacee_info=dead_successor_info,
                    replacee_was_tail=dead_succ_was_tail,
                )

                if self.llm.output_layer_loaded:
                    self.chain.update_load_output_layer(False)

                self._connect_to_successor()

                if self.chain.get_all_layers_loaded():
                    self.chain.update_chain_status(ChainStatus.READY)
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
                                avail_mem=avail_mem,
                                avail_vram=avail_vram,
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
                self.chain.update_all_layers_loaded()
                self.chain.update_chain_status(ChainStatus.UNREADY)

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
        self.chain.dht.store(ALL_LAYERS_KEY, False, EXPIRATION_S)
        self.chain.update_chain_status(ChainStatus.TAKEOVER)

        layers_to_takeover = weak_node_info.get("layers")
        weak_node_output_layer_loaded = weak_node_info.get("output_layer_loaded", False)

        weak_node_was_tail = self.chain.node_is_tail(weak_node_info.get("id"))
        weak_node_was_head = self.chain.node_is_head(weak_node_info.get("id"))
        logger.info(
            f"Backup Node: {self.chain.node_id} attempting to takeover layers {layers_to_takeover}."
        )

        if self._can_load(layers=layers_to_takeover, output_layer=weak_node_output_layer_loaded):
            
            self._reload_llm(layers_to_takeover, load_output_layer=weak_node_output_layer_loaded)
            
            self.chain.repair(
                layers_to_takeover,
                replacement_info=self.chain.get_self_info(),
                replacement_load_output_layer=weak_node_output_layer_loaded,
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
                f"Backup Node: {self.chain.node_id} can not load layers {layers_to_takeover} of the weak node."
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
            server_node.chain.republish_keys()
            server_node._update_memory_usage()
            server_node.chain.update_all_layers_loaded()

    def _chain_health_monitor_task(server_node: Server):
        """Backgroud task to check on the node's successor status"""
        while True:
            time.sleep(HEARTBEAT_INTERVAL_S)
            if server_node.chain.is_backup():
                # if the TAIL doesn't have the final layer loaded the backup node can join the chain
                if (
                    not server_node.chain.get_all_layers_loaded()
                    and server_node.chain.get_chain_status() == ChainStatus.UNREADY
                ):
                    max_num_layers = server_node._mem_to_num_layers()
                    if max_num_layers > 0:
                        # Join chain
                        server_node.chain.join_chain(
                            self_info=server_node.chain.get_self_info(),
                            max_num_layers=server_node._mem_to_num_layers(),
                            max_num_params=server_node._mem_to_num_params(),
                            num_total_layers=server_node.config.get("num_hidden_layers"),
                            num_total_params=server_node.config.get("num_total_params"),
                            transformer_layer_params=server_node.config.get("transformer_layer_params"),
                            final_output_params=server_node.config.get("final_output_params")
                        )
                        server_node._load_llm()

                # Opportunistic Takeover
                weak_node_info, predecessor_info = server_node.chain.evaluate_takeover_eligibility()

                if weak_node_info and server_node.chain.get_chain_status() == ChainStatus.READY:
                    server_node.opportunistic_takeover(weak_node_info, predecessor_info)

            elif server_node.chain.is_tail():
                continue

            # Perform the health check on the successor
            if server_node.chain.get_chain_status() in (ChainStatus.READY, ChainStatus.UNREADY):
                try:
                    server_node._connect_to_successor()
                    if server_node.successor_stub is not None:
                        server_node.successor_stub.Check(nodeservice_pb2.Empty(), timeout=2)
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

    # Load the server's assigned layers
    server_node._load_llm()

    # Profile the node to get processing rate measurements
    server_node._profile_node(dummy_seq_length=50, profiling_runs=300, profiling_duration_s=PROFILING_DURATION)

    grpc_server.wait_for_termination()


if __name__ == "__main__":
    serve()
