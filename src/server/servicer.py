import logging
from typing import TYPE_CHECKING

import grpc
import torch
import time

from core.remote import nodeservice_pb2, nodeservice_pb2_grpc
from core.remote.serialization import message_to_tensor

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
        input_tensor = message_to_tensor(request)

        # Extract metadata
        max_returned_tokens = request.max_returned_tokens
        seq_length = request.seq_length if request.HasField("seq_length") else None
        input_pos_val = request.input_pos if request.HasField("input_pos") else None
        input_pos = torch.tensor([input_pos_val]) if input_pos_val is not None else None

        partial_rate = request.partial_rate
        response_address = request.response_address

        # Run the inference logic
        ack_response = self.server_node.run_local_layers(
            input_tensor, 
            max_returned_tokens, 
            seq_length, input_pos, 
            partial_rate,
            response_address
        )

        end = time.perf_counter()

        total_grpc_time = end - start
        self.server_node.grpc_overhead = total_grpc_time - self.server_node.inference_delay - ack_response.processing_time

        logger.info(f"GPRC Overhead: {self.server_node.grpc_overhead:.6f}s")

        ack_response.processing_time = total_grpc_time

        # Update inference delay and grpc overhead on the dht
        self.server_node.chain.update_inference_delay(self.server_node.inference_delay)
        self.server_node.chain.update_grpc_overhead(self.server_node.grpc_overhead)


        return ack_response

    def Check(self, request, context):
        """If the server is running it will return an Empty response"""
        return nodeservice_pb2.Empty()
    
    def Reallocate(self, request, context):
        total_rate = request.total_rate
        start_layer_index = request.start_layer_index

        self.server_node.reallocate_layers(total_rate, start_layer_index)
        return nodeservice_pb2.Empty()
    
    def LoadLayers(self, request, context):
        # start_idx = request.start
        # end_idx = request.end

        self.server_node._load_llm()
        return nodeservice_pb2.Empty()
