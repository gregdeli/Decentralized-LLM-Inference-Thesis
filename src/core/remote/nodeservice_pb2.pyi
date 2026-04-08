from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class MultiaddrResponse(_message.Message):
    __slots__ = ("multiaddr",)
    MULTIADDR_FIELD_NUMBER: _ClassVar[int]
    multiaddr: str
    def __init__(self, multiaddr: _Optional[str] = ...) -> None: ...

class InferenceRequest(_message.Message):
    __slots__ = ("tensor_data", "tensor_shape", "dtype", "max_returned_tokens", "seq_length", "input_pos", "partial_rate", "response_address", "block_scales")
    TENSOR_DATA_FIELD_NUMBER: _ClassVar[int]
    TENSOR_SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    MAX_RETURNED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    SEQ_LENGTH_FIELD_NUMBER: _ClassVar[int]
    INPUT_POS_FIELD_NUMBER: _ClassVar[int]
    PARTIAL_RATE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    BLOCK_SCALES_FIELD_NUMBER: _ClassVar[int]
    tensor_data: bytes
    tensor_shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    max_returned_tokens: int
    seq_length: int
    input_pos: int
    partial_rate: float
    response_address: str
    block_scales: bytes
    def __init__(self, tensor_data: _Optional[bytes] = ..., tensor_shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., max_returned_tokens: _Optional[int] = ..., seq_length: _Optional[int] = ..., input_pos: _Optional[int] = ..., partial_rate: _Optional[float] = ..., response_address: _Optional[str] = ..., block_scales: _Optional[bytes] = ...) -> None: ...

class InferenceResponse(_message.Message):
    __slots__ = ("tensor_data", "tensor_shape", "dtype", "error_message", "total_rate", "processing_time", "block_scales")
    TENSOR_DATA_FIELD_NUMBER: _ClassVar[int]
    TENSOR_SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_RATE_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_TIME_FIELD_NUMBER: _ClassVar[int]
    BLOCK_SCALES_FIELD_NUMBER: _ClassVar[int]
    tensor_data: bytes
    tensor_shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    error_message: str
    total_rate: float
    processing_time: float
    block_scales: bytes
    def __init__(self, tensor_data: _Optional[bytes] = ..., tensor_shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., error_message: _Optional[str] = ..., total_rate: _Optional[float] = ..., processing_time: _Optional[float] = ..., block_scales: _Optional[bytes] = ...) -> None: ...

class ReallocateRequest(_message.Message):
    __slots__ = ("total_rate", "start_layer_index")
    TOTAL_RATE_FIELD_NUMBER: _ClassVar[int]
    START_LAYER_INDEX_FIELD_NUMBER: _ClassVar[int]
    total_rate: float
    start_layer_index: int
    def __init__(self, total_rate: _Optional[float] = ..., start_layer_index: _Optional[int] = ...) -> None: ...
