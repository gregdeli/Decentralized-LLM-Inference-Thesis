import logging
from typing import TYPE_CHECKING

import grpc
import torch

from core.remote import nodeservice_pb2, nodeservice_pb2_grpc

if TYPE_CHECKING:
    from .client import Client

logger = logging.getLogger(__name__)

class ClientServicer(nodeservice_pb2_grpc.ClientServiceServicer):
    def __init__(self, client_node: "Client"):
        self.client_node = client_node

    def ReceiveResponse(self, request, context):
        self.client_node.inference_response = request
        self.client_node.inference_response_event.set()
        return nodeservice_pb2.Empty()