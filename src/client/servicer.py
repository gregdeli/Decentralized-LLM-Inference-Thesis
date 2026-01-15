import logging
from typing import TYPE_CHECKING

from core.remote import nodeservice_pb2, nodeservice_pb2_grpc
from core.remote.serialization import message_to_tensor

if TYPE_CHECKING:
    from .client import Client

logger = logging.getLogger(__name__)

class ClientServicer(nodeservice_pb2_grpc.ClientServiceServicer):
    def __init__(self, client_node: "Client"):
        self.client_node = client_node

    def ReceiveResponse(self, request, context):
        # response = self.inference_response

        # Capture TOTAL RATE
        if request.total_rate > 0:
            self.client_node.total_rate = request.total_rate

        self.client_node.final_activations = message_to_tensor(request)

        # self.client_node.inference_response = request
        self.client_node.inference_response_event.set()
        return nodeservice_pb2.Empty()