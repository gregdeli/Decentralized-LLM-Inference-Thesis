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

def _build_dynamic_qmap() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs the 256-value non-linear dynamic quantization map.
    The dynamic format consists of 1 sign bit, an exponent of zeros
    terminated by 1, the remainining bits acting as the fraction
    """
    qmap = torch.zeros(256, dtype=torch.bfloat16)
    for i in range(256):
        binary = format(i, '08b')
        sign = -1 if binary[0] == '1' else 1

        indicator_idx = binary.find('1', 1)
        if indicator_idx == -1:
            qmap[i] = 0.0
        else:
            E = indicator_idx - 1 # exponent
            fraction_str = binary[indicator_idx + 1:]
            L = len(fraction_str) # num fraction bits
            F = int(fraction_str, 2) if L > 0 else 1.0 # convert binary fraction to binary
            mantissa = (F / (2.0 ** L - 1)) if L > 0 else F
            
            val = (10.0 ** -E) * mantissa
            qmap[i] = sign * val

    # Normalize to [-1.0, 1.0] to match absmax division
    qmap = qmap / qmap.abs().max()

    # Sort for binary search 
    qmap_sorted, _ = torch.sort(qmap)

    midpoints = (qmap_sorted[:-1] + qmap_sorted[1:]) / 2.0

    return qmap_sorted, midpoints



QMAP_SORTED, QMAP_MIDPOINTS = _build_dynamic_qmap()



def tensor_to_request(
    tensor: torch.Tensor,
    max_returned_tokens: int,
    seq_length: Optional[int],
    input_pos: Optional[int],
) -> nodeservice_pb2.InferenceRequest:
    """Serializes a tensor and metadata into an InferenceRequest."""
    quantized_indices, absmax = quantize_blockwise(tensor)

    quantized_indices_bytes = quantized_indices.numpy().tobytes()
    tensor_shape = list(tensor.shape)
    dtype = DTYPE_MAP[tensor.dtype]

    view_dtype = torch.int16 if absmax.element_size() == 2 else absmax.dtype
    block_scales = absmax.view(view_dtype).numpy().tobytes()

    # Create the request with explicit arguments
    request_args = {
        "tensor_data": quantized_indices_bytes,
        "tensor_shape": tensor_shape,
        "dtype": dtype,
        "max_returned_tokens": max_returned_tokens,
        "seq_length": seq_length,
        "input_pos": input_pos,
        "block_scales": block_scales,
    }

    return nodeservice_pb2.InferenceRequest(**request_args)


def tensor_to_response(tensor: torch.Tensor) -> nodeservice_pb2.InferenceResponse:
    tensor_shape = list(tensor.shape)
    dtype = DTYPE_MAP[tensor.dtype]

    view_dtype = torch.int16 if tensor.element_size() == 2 else tensor.dtype
    tensor_data = tensor.view(view_dtype).numpy().tobytes()

    response = {
        "tensor_data": tensor_data,
        "tensor_shape": tensor_shape,
        "dtype": dtype,
    }

    return nodeservice_pb2.InferenceResponse(**response)


def request_to_tensor(request: nodeservice_pb2.InferenceRequest) -> torch.Tensor:
    """Deserializes an InferenceRequest into a tensor"""
    shape = tuple(request.tensor_shape)

    dtype_str = request.dtype
    torch_dtype = INV_DTYPE_MAP[dtype_str]

    is_2byte = torch_dtype.itemsize == 2
    np_dtype = np.int16 if is_2byte else getattr(np, dtype_str)

    # Load the quantized uint8 data
    np_quantized_indices = np.frombuffer(request.tensor_data, dtype=np.uint8)
    quantized_indices = torch.from_numpy(np_quantized_indices).view(-1, BLOCK_SIZE)

    # Load the block scales
    np_scales = np.frombuffer(request.block_scales, dtype=np_dtype)
    absmax_tensor = torch.from_numpy(np_scales).view(-1, 1).view(torch_dtype)

    # Dequantize
    tensor = dequantize_blockwise(
        quantized_indices=quantized_indices,
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
    """Compresses a tensor to uint8 using  Dynamic Block-wise Quantization (Dettmers, Tim, et al.)"""
    flat_tensor = tensor.flatten()

    # Reshape into blocks
    blocks = flat_tensor.view(-1, block_size)

    # Compute absmax for each block
    absmax = blocks.abs().max(dim=-1, keepdim=True).values
    absmax = torch.clamp(absmax, min=1e-8)  # Prevent division by zero

    # Normalize to [-1, 1]
    normalized_blocks = blocks / absmax

    # Nearest neighbor search via midpoints
    midpoints = QMAP_MIDPOINTS.to(tensor.device)
    indices = torch.bucketize(normalized_blocks.contiguous(), midpoints)

    quantized_indices = indices.to(torch.uint8)

    return quantized_indices, absmax


def dequantize_blockwise(
    quantized_indices: torch.Tensor,
    absmax: torch.Tensor,
    original_shape: tuple,
    target_dtype: torch.dtype,
):
    """Restores the uint8 back to the target float type using the dynamic QMAP"""
    # Use int64 for safe indexing
    indices = quantized_indices.to(torch.int64) 

    # Lookup values in the pre-computed dynamic tree map
    qmap = QMAP_SORTED.to(device=quantized_indices.device, dtype=target_dtype)

    flat_indices = indices.flatten()
    dequantized_normalized = qmap[flat_indices].view(indices.shape)

    # Re-apply scales
    dequantized = dequantized_normalized * absmax.to(target_dtype)

    # Flatten
    flat_tensor = dequantized.flatten()

    return flat_tensor.view(original_shape)
