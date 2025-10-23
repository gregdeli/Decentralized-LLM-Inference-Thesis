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
        # Create the successor stub if it doesn't exist
        # if self.server_node.successor_stub is None:
        #     if self.server_node.chain.is_tail():
        #         self.server_node.successor_stub = None
        #     else:
        #         successor_addr = self.server_node.chain.get_successor_address()
        #         if not successor_addr:
        #             context.abort(grpc.StatusCode.INTERNAL, "Successor not found for a non-tail node.")
        #             return nodeservice_pb2.InferenceResponse()

        #         try:
        #             channel = grpc.insecure_channel(successor_addr)
        #             grpc.channel_ready_future(channel).result(timeout=10)

        #             self.server_node.successor_stub = nodeservice_pb2_grpc.NodeServiceStub(channel)
        #             logger.info(f"Connection to successor: {successor_addr} established.")  # Debugging
        #         except grpc.FutureTimeoutError:
        #             logger.error(f"Connection to {successor_addr} timed out.")
        #         except grpc.RpcError as e:
        #             logger.error(f"A gRPC error occurred while connecting: {e.code().name}")

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
    
    def Check(self, request, context):
        """ If the server is running it will return an Empty response"""
        return nodeservice_pb2.Empty()