import torch
import numpy as np
from . import nodeservice_pb2
from typing import Optional, Union, Tuple

BLOCK_SIZE = 1024  # A standard block size used for block-wise quantization

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
    quantized_blocks, absmax = quantize_blockwise(tensor)

    tensor_data = quantized_blocks.numpy().tobytes()
    tensor_shape = list(tensor.shape)
    dtype = DTYPE_MAP[tensor.dtype]

    view_dtype = torch.int16 if absmax.element_size() == 2 else absmax.dtype
    block_scales = absmax.view(view_dtype).numpy().tobytes()

    # Create the request with explicit arguments
    request_args = {
        "tensor_data": tensor_data,
        "tensor_shape": tensor_shape,
        "dtype": dtype,
        "max_returned_tokens": max_returned_tokens,
        "seq_length": seq_length,
        "input_pos": input_pos,
        "block_scales": block_scales,
    }

    return nodeservice_pb2.InferenceRequest(**request_args)


def tensor_to_response(tensor: torch.Tensor) -> nodeservice_pb2.InferenceResponse:
    # quantized_blocks, absmax = quantize_blockwise(tensor)
    # tensor_data = quantized_blocks.numpy().tobytes()
    tensor_shape = list(tensor.shape)
    dtype = DTYPE_MAP[tensor.dtype]

    # view_dtype = torch.int16 if absmax.element_size() == 2 else absmax.dtype
    # block_scales = absmax.view(view_dtype).numpy().tobytes()

    view_dtype = torch.int16 if tensor.element_size() == 2 else tensor.dtype
    tensor_data = tensor.view(view_dtype).numpy().tobytes()

    response = {
        "tensor_data": tensor_data,
        "tensor_shape": tensor_shape,
        "dtype": dtype,
        # "block_scales": block_scales,
    }

    return nodeservice_pb2.InferenceResponse(**response)


def request_to_tensor(request: nodeservice_pb2.InferenceRequest) -> torch.Tensor:
    """Deserializes an InferenceRequest into a tensor"""
    shape = tuple(request.tensor_shape)

    dtype_str = request.dtype
    torch_dtype = INV_DTYPE_MAP[dtype_str]

    is_2byte = torch_dtype.itemsize == 2
    np_dtype = np.int16 if is_2byte else getattr(np, dtype_str)

    # Load the quantized int8 data
    np_quantized = np.frombuffer(request.tensor_data, dtype=np.int8)
    quantized_tensor = torch.from_numpy(np_quantized).view(-1, BLOCK_SIZE)

    # Load the block scales
    np_scales = np.frombuffer(request.block_scales, dtype=np_dtype)
    absmax_tensor = torch.from_numpy(np_scales).view(-1, 1).view(torch_dtype)

    # Dequantize
    tensor = dequantize_blockwise(
        quantized_blocks=quantized_tensor,
        absmax=absmax_tensor,
        original_shape=shape,
        target_dtype=torch_dtype,
    )

    return tensor

def response_to_tensor(response: nodeservice_pb2.InferenceResponse) -> torch.Tensor:
    """Deserializes an InferenceResponse into a tensor"""
    tensor_data = bytearray(response.tensor_data)

    shape = tuple(response.tensor_shape)

    dtype_str = response.dtype
    torch_dtype = INV_DTYPE_MAP[dtype_str]

    tensor = torch.frombuffer(tensor_data, dtype=torch_dtype)
    tensor = tensor.reshape(shape)

    return tensor



def quantize_blockwise(
    tensor: torch.Tensor, block_size: int = BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compresses a tensor to int8 using dynamic block-wise quantization"""
    flat_tensor = tensor.flatten()

    # Reshape into blocks
    blocks = flat_tensor.view(-1, block_size)

    # Compute absmax for each block
    absmax = blocks.abs().max(dim=-1, keepdim=True).values
    absmax = torch.clamp(absmax, min=1e-8)  # Prevent division by zero

    # Quantize to 8-bit integer [-127, 127]
    quantized_blocks = torch.round((blocks / absmax) * 127.0).to(torch.int8)

    return quantized_blocks, absmax


def dequantize_blockwise(
    quantized_blocks: torch.Tensor,
    absmax: torch.Tensor,
    original_shape: tuple,
    target_dtype: torch.dtype,
):
    """Restores the int8 back to the target float type"""
    dequantized = quantized_blocks.to(target_dtype)

    # Re-apply the scale
    dequantized = (dequantized / 127.0) * absmax

    # Flatten
    flat_tensor = dequantized.flatten()

    return flat_tensor.view(original_shape)
