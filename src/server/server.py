"""Main server application logic"""

import grpc
import os
from concurrent import futures
import torch
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple

from core.llm_loader import LLM
from core.remote import inference_pb2, inference_pb2_grpc
from core.remote.serialization import *


class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def __init__(self, server_node: "Server"):
        self.server_node = server_node

    def RunLayers(self, request, context):
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
        layers_start_idx: int = 0,
        is_head: bool = False,
        successor_addr: Optional[str] = None,
        time_it: bool = False,
    ) -> None:
        self.llm = LLM.load(
            model_path,
            is_client=False,
            num_layers=num_layers,
            layers_start_idx=layers_start_idx,
            time_it=time_it,
        )
        self.model = self.llm.model
        self.is_head = is_head
        self.is_tail = self.llm.layers_loaded[1] == (self.llm.config["num_hidden_layers"] - 1)

        self.successor_addr = successor_addr
        self.successor_stub = None
        if self.successor_addr:
            channel = grpc.insecure_channel(self.successor_addr)
            self.successor_stub = inference_pb2_grpc.InferenceStub(channel)

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
            device = self.llm.preprocessor.device
            # Na allaksw to batch_size otan kanw batched inference
            self.model.set_kv_cache(batch_size=1, max_seq_length=max_returned_tokens, device=device)
            self.llm.kv_cache_initialized = True

        h = self.model.forward_server(input_tensor, seq_length, input_pos)

        if self.is_tail:
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
    layers_start_idx = int(os.getenv("LAYERS_START_IDX"))
    is_head_str = os.getenv("IS_HEAD", "False")
    is_head = is_head_str.lower() in ("true", "1")
    successor_addr = os.getenv("SUCCESSOR_ADDR", None)
    port = os.getenv("PORT", "50051")

    server_node = Server(
        model_path=model_path,
        num_layers=num_layers,
        layers_start_idx=layers_start_idx,
        is_head=is_head,
        successor_addr=successor_addr,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(server_node), server)
    server.add_insecure_port(f"[::]:{port}")
    print(f"Server listening on port {port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
