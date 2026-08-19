import logging
from typing import TYPE_CHECKING

import grpc
import torch
import time
import json
import threading
import concurrent.futures
import gc
import ctypes

from core.remote import nodeservice_pb2, nodeservice_pb2_grpc
from core.remote.utils import create_grpc_channel
from core.remote.serialization import request_to_tensor
from core.constants import GLOBAL_MAX_SEQ_LEN, GRPC_MAX_WORKERS

from core.p2p.chain_manager import ChainStatus

if TYPE_CHECKING:
    from .server import Server

logger = logging.getLogger(__name__)


class NodeServicer(nodeservice_pb2_grpc.NodeServiceServicer):
    def __init__(self, server_node: "Server"):
        self.server_node = server_node

        self.inference_executor = concurrent.futures.ThreadPoolExecutor(max_workers=GRPC_MAX_WORKERS)

    def GetPeerMultiaddr(self, request, context):
        visible_maddrs = self.server_node.dht.get_visible_maddrs()
        if not visible_maddrs:
            context.abort(grpc.StatusCode.UNAVAILABLE, "P2P address not yet available.")

        return nodeservice_pb2.MultiaddrResponse(multiaddr=str(visible_maddrs[0]))

    def RunLayers(self, request, context):
        # logger.info(f"Begin RunLayers handling from ({request.response_address})...")
        # Memory Debug
        logger.debug(f"Memory Usage: {self.server_node.memory_usage_mb} MB")
        logger.debug(f"VRAM Usage: {self.server_node.vram_usage_mb} MB")

        # Deserialize the incoming request to a tensor
        start = time.perf_counter()
        input_tensor = request_to_tensor(request, quantize_flag=self.server_node.quantize_flag)

        # Extract metadata
        max_returned_tokens = request.max_returned_tokens
        seq_length = request.seq_length if request.HasField("seq_length") else None
        input_pos_val = request.input_pos if request.HasField("input_pos") else None
        if seq_length > 1:
            input_pos = torch.arange(
                input_pos_val, input_pos_val + seq_length, device=self.server_node.device
            )
        else:
            input_pos = torch.tensor([input_pos_val])

        response_address = request.response_address

        # Ensure inputs are on the same device as the model
        if input_tensor.device != self.server_node.device:
            input_tensor = input_tensor.to(self.server_node.device)

        if input_pos is not None and input_pos.device != self.server_node.device:
            input_pos = input_pos.to(self.server_node.device)

        deserialization_delay = time.perf_counter() - start
        logger.debug(f"Deserialization Delay: {deserialization_delay:.12f}")
        # self.server_node.chain.update_deserialization_delay(deserialization_delay)

        # threading.Thread(
        #     target=self.server_node.run_local_layers,
        #     args=(input_tensor, max_returned_tokens, seq_length, input_pos, response_address),
        #     daemon=True,
        # ).start()

        submit_time = time.perf_counter()
        self.inference_executor.submit(
            self.server_node.run_local_layers,
            input_tensor,
            max_returned_tokens,
            seq_length, 
            input_pos, 
            response_address,
            submit_time
        )

        # self.server_node.run_local_layers(input_tensor, max_returned_tokens, seq_length, input_pos, response_address)

        # logger.info(f"End RunLayers handling from ({request.response_address})...")
        return nodeservice_pb2.Empty(deserialization_delay=deserialization_delay)

    def Check(self, request, context):
        """If the server is running it will return an Empty response"""
        return nodeservice_pb2.Empty()

    def Reallocate(self, request, context):
        total_rate = request.total_rate
        start_layer_index = request.start_layer_index
        load_max = request.load_max
        predecessor_info = (
            json.loads(request.predecessor_info.decode("utf-8"))
            if request.predecessor_info
            else None
        )

        response = self.server_node.reallocate_layers(
            total_rate, start_layer_index, load_max, predecessor_info
        )

        return response

    def TriggerReallocation(self, request, context):
        self.server_node.trigger_reallocation()

        return nodeservice_pb2.Empty()

    def UpdateSuccessorStub(self, request, context):
        self.server_node._connect_to_successor()
        return nodeservice_pb2.Empty()

    def LoadLayers(self, request, context):
        self.server_node._unload_llm()
        self.server_node._load_llm()
        self.server_node._connect_to_successor()
        return nodeservice_pb2.Empty()

    def UnloadLayers(self, request, context):
        self.server_node._unload_llm()
        return nodeservice_pb2.Empty()

    def RemoveClientKVCache(self, request, context):
        # Free the client's cache memory
        logger.info(f"Removing client's {request.client_id} KV Cache...")
        self.server_node.model.remove_client_cache(request.client_id)

        # Reallocate the remaining caches
        logger.info(f"Reallocating the remaining KV Caches...")
        self.server_node.model.reallocate_caches(
            batch_size=1,
            max_seq_length=request.new_max_seq_length,
            device=self.server_node.device,
            dtype=self.server_node.llm.dtype,
        )

        # Release freed memory
        gc.collect()
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception as e:
            logger.warning(f"Failed to trim memory: {e}")

        # Update memory usage
        self.server_node._update_memory_usage()

        # Forward to successor
        request = nodeservice_pb2.RemoveClientKVCacheRequest(
            client_id=request.client_id, new_max_seq_length=request.new_max_seq_length
        )
        if not self.server_node.chain.is_tail() and self.server_node.successor_stub:
            try:
                # self.server_node._connect_to_successor()
                self.server_node.successor_stub.RemoveClientKVCache(request)
                return nodeservice_pb2.Empty()
            except grpc.RpcError as e:
                logger.error(f"Failed to forward Client Removal Request to successor: {e}")

        return nodeservice_pb2.Empty()

    def JoinChain(self, request, context):
        previous_tail_info = self.server_node.chain.join_chain(
            self_info=self.server_node.chain.get_self_info(),
            max_num_layers=self.server_node._mem_to_num_layers(),
            output_layer_memory_tle=self.server_node.output_layer_memory_tle,
            num_total_layers=self.server_node.config.get("num_hidden_layers"),
            num_total_params=self.server_node.config.get("num_total_params"),
        )

        # Update the previous TAIL's successor stub
        if previous_tail_info is not None:
            try:
                logger.info(f"Updating the previous tail's successor stub...")
                channel = create_grpc_channel(previous_tail_info.get("address"))
                previous_tail_stub = nodeservice_pb2_grpc.NodeServiceStub(channel)
                previous_tail_stub.UpdateSuccessorStub(nodeservice_pb2.Empty())
            except grpc.RpcError as e:
                logger.error(
                    f"A gRPC error occurred while connecting to {previous_tail_info.get('address')}: {e.code().name}"
                )

        # threading.Thread(target=self.server_node._load_llm, daemon=True).start()
        self.server_node._load_llm()
        # self.server_node._connect_to_successor()

        return nodeservice_pb2.Empty()

    def UpdateChainStatus(self, request, context):
        status_str = request.status
        chain_status = ChainStatus(status_str)
        self.server_node.chain.update_chain_status(chain_status)

        return nodeservice_pb2.Empty()
