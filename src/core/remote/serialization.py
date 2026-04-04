import torch
import numpy as np
from . import nodeservice_pb2
from typing import Optional, Union

# A mapping from PyTorch dtypes to string representations
DTYPE_MAP = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.int64: "int64",
}
INV_DTYPE_MAP = {dtype_str: dtype for dtype, dtype_str in DTYPE_MAP.items()}


def tensor_to_request(
    tensor: torch.Tensor,
    max_returned_tokens: int,
    seq_length: Optional[int],
    input_pos: Optional[int],
) -> nodeservice_pb2.InferenceRequest:
    """Serializes a tensor and metadata into an InferenceRequest."""
    view_dtype = torch.int16 if tensor.element_size() == 2 else tensor.dtype
    tensor_data = tensor.view(view_dtype).numpy().tobytes()
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


def tensor_to_response(tensor: torch.Tensor) -> nodeservice_pb2.InferenceResponse:
    view_dtype = torch.int16 if tensor.element_size() == 2 else tensor.dtype
    tensor_data = tensor.view(view_dtype).numpy().tobytes()
    tensor_shape = list(tensor.shape)
    dtype = DTYPE_MAP[tensor.dtype]

    response = {
        "tensor_data": tensor_data,
        "tensor_shape": tensor_shape,
        "dtype": dtype,
    }

    return nodeservice_pb2.InferenceResponse(**response)


def message_to_tensor(message: Union[nodeservice_pb2.InferenceRequest, nodeservice_pb2.InferenceResponse]) -> torch.Tensor:
    """Deserializes an InferenceRequest or an InferenceResponse into a tensor."""
    shape = tuple(message.tensor_shape)
    dtype_str = message.dtype

    torch_dtype = INV_DTYPE_MAP[dtype_str]
    is_2byte = torch_dtype.itemsize == 2
    np_dtype = np.int16 if is_2byte else getattr(np, dtype_str)

    np_array = np.frombuffer(message.tensor_data, dtype=np_dtype)
    tensor = torch.from_numpy(np_array).reshape(shape)
    return tensor.view(torch_dtype) if is_2byte else tensor
