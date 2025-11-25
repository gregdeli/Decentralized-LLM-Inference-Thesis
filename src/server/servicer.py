import logging
from typing import TYPE_CHECKING

import grpc
import torch

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

        return nodeservice_pb2.MultiaddrResponse(multiaddr=str(visible_maddrs[1]))

    def RunLayers(self, request, context):
        # Deserialize the incoming request to a tensor
        input_tensor = message_to_tensor(request)

        # Extract metadata
        max_returned_tokens = request.max_returned_tokens
        seq_length = request.seq_length if request.HasField("seq_length") else None
        input_pos_val = request.input_pos if request.HasField("input_pos") else None
        input_pos = torch.tensor([input_pos_val]) if input_pos_val is not None else None

        partial_rate = request.partial_rate

        # Run the actual inference logic
        final_layer_response = self.server_node.run_local_layers(
            input_tensor, 
            max_returned_tokens, 
            seq_length, input_pos, 
            partial_rate
        )

        return final_layer_response

    def Check(self, request, context):
        """If the server is running it will return an Empty response"""
        return nodeservice_pb2.Empty()
    
    def Reallocate(self, request, context):
        total_rate = request.total_rate
        start_layer_index = request.start_layer_index

        self.server_node.reallocate_layers(total_rate, start_layer_index)
        return nodeservice_pb2.Empty()
