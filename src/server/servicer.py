import logging
from typing import TYPE_CHECKING

import grpc
import torch
import time
import json

from core.remote import nodeservice_pb2, nodeservice_pb2_grpc
from core.remote.serialization import request_to_tensor
from core.constants import GLOBAL_MAX_SEQ_LEN

if TYPE_CHECKING:
    from .server import Server

logger = logging.getLogger(__name__)


class NodeServicer(nodeservice_pb2_grpc.NodeServiceServicer):
    def __init__(self, server_node: "Server"):
        self.server_node = server_node

    def GetPeerMultiaddr(self, request, context):
        visible_maddrs = self.server_node.dht.get_visible_maddrs()
        if not visible_maddrs:
            context.abort(grpc.StatusCode.UNAVAILABLE, "P2P address not yet available.")

        return nodeservice_pb2.MultiaddrResponse(multiaddr=str(visible_maddrs[0]))

    def RunLayers(self, request, context):
        start = time.perf_counter()

        # Deserialize the incoming request to a tensor
        input_tensor = request_to_tensor(request)

        # Extract metadata
        max_returned_tokens = request.max_returned_tokens
        seq_length = request.seq_length if request.HasField("seq_length") else None
        input_pos_val = request.input_pos if request.HasField("input_pos") else None
        input_pos = torch.tensor([input_pos_val]) if input_pos_val is not None else None

        # partial_rate = request.partial_rate
        response_address = request.response_address

        # Run the inference logic
        ack_response = self.server_node.run_local_layers(
            input_tensor, 
            max_returned_tokens, 
            seq_length, 
            input_pos, 
            response_address
        )

        end = time.perf_counter()

        total_grpc_time = end - start
        self.server_node.grpc_overhead = (
            total_grpc_time - self.server_node.inference_delay - ack_response.processing_time
        )

        logger.info(f"GPRC Overhead: {self.server_node.grpc_overhead:.6f}s")

        ack_response.processing_time = total_grpc_time

        # Update processing rate, inference delay and grpc overhead on the dht
        self.server_node.chain.update_processing_rate(self.server_node.processing_rate)
        self.server_node.chain.update_inference_delay(self.server_node.inference_delay)
        self.server_node.chain.update_grpc_overhead(self.server_node.grpc_overhead)

        return ack_response

    def Check(self, request, context):
        """If the server is running it will return an Empty response"""
        return nodeservice_pb2.Empty()

    def Reallocate(self, request, context):
        total_rate = request.total_rate
        start_layer_index = request.start_layer_index
        predecessor_info = (
            json.loads(request.predecessor_info.decode("utf-8"))
            if request.predecessor_info
            else None
        )

        self.server_node.reallocate_layers(total_rate, start_layer_index, predecessor_info)
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
            dtype=self.server_node.llm.dtype
        )

        self.server_node._update_memory_usage()

        # Forward to successor
        request = nodeservice_pb2.RemoveClientKVCacheRequest(
            client_id=request.client_id,
            new_max_seq_length=request.new_max_seq_length
        )
        if not self.server_node.chain.is_tail() and self.server_node.successor_stub:
            try:
                # self.server_node._connect_to_successor()
                self.server_node.successor_stub.Reallocate(request)
                return nodeservice_pb2.Empty()
            except grpc.RpcError as e:
                logger.error(f"Failed to forward Client Removal Request to successor: {e}")

        return nodeservice_pb2.Empty()
