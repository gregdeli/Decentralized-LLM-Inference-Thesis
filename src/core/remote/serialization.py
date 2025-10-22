import torch
import numpy as np
from . import nodeservice_pb2
from typing import Optional, Union

# A mapping from PyTorch dtypes to string representations
DTYPE_MAP = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.int64: "int64",
}
# INV_DTYPE_MAP = {v: k for k, v in DTYPE_MAP.items()}


def tensor_to_request(
    tensor: torch.Tensor,
    max_returned_tokens: int,
    seq_length: Optional[int],
    input_pos: Optional[int],
) -> nodeservice_pb2.InferenceRequest:
    """Serializes a tensor and metadata into an InferenceRequest."""
    tensor_data = tensor.numpy().tobytes()
    tensor_shape = list(tensor.shape)
    dtype = DTYPE_MAP[tensor.dtype]

    # Create the request with explicit arguments
    request_args = {
        "tensor_data": tensor_data,
        "tensor_shape": tensor_shape,
        "dtype": dtype,
        "max_returned_tokens": max_returned_tokens,
        "seq_length": seq_length,
        "input_pos": input_pos,
    }
    # if seq_length is not None:
    #     request_args["seq_length"] = seq_length
    # if input_pos is not None:
    #     request_args["input_pos"] = input_pos

    return nodeservice_pb2.InferenceRequest(**request_args)


def tensor_to_response(tensor: torch.Tensor) -> nodeservice_pb2.InferenceRequest:
    tensor_data = tensor.numpy().tobytes()
    tensor_shape = list(tensor.shape)
    dtype = DTYPE_MAP[tensor.dtype]

    response = {
        "tensor_data": tensor_data,
        "tensor_shape": tensor_shape,
        "dtype": dtype,
    }

    return nodeservice_pb2.InferenceRequest(**response)


def message_to_tensor(response: Union[nodeservice_pb2.InferenceResponse, nodeservice_pb2.InferenceResponse]) -> torch.Tensor:
    """Deserializes an InferenceRequest or an InferenceResponse into a tensor."""
    shape = tuple(response.tensor_shape)
    dtype_str = response.dtype

    np_array = np.frombuffer(response.tensor_data, dtype=getattr(np, dtype_str)).copy()
    tensor = torch.from_numpy(np_array).reshape(shape)
    return tensor
