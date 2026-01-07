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
    __slots__ = ("tensor_data", "tensor_shape", "dtype", "max_returned_tokens", "seq_length", "input_pos", "partial_rate", "response_address")
    TENSOR_DATA_FIELD_NUMBER: _ClassVar[int]
    TENSOR_SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    MAX_RETURNED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    SEQ_LENGTH_FIELD_NUMBER: _ClassVar[int]
    INPUT_POS_FIELD_NUMBER: _ClassVar[int]
    PARTIAL_RATE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    tensor_data: bytes
    tensor_shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    max_returned_tokens: int
    seq_length: int
    input_pos: int
    partial_rate: float
    response_address: str
    def __init__(self, tensor_data: _Optional[bytes] = ..., tensor_shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., max_returned_tokens: _Optional[int] = ..., seq_length: _Optional[int] = ..., input_pos: _Optional[int] = ..., partial_rate: _Optional[float] = ..., response_address: _Optional[str] = ...) -> None: ...

class InferenceResponse(_message.Message):
    __slots__ = ("tensor_data", "tensor_shape", "dtype", "error_message", "total_rate")
    TENSOR_DATA_FIELD_NUMBER: _ClassVar[int]
    TENSOR_SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_RATE_FIELD_NUMBER: _ClassVar[int]
    tensor_data: bytes
    tensor_shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    error_message: str
    total_rate: float
    def __init__(self, tensor_data: _Optional[bytes] = ..., tensor_shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., error_message: _Optional[str] = ..., total_rate: _Optional[float] = ...) -> None: ...

class ReallocateRequest(_message.Message):
    __slots__ = ("total_rate", "start_layer_index")
    TOTAL_RATE_FIELD_NUMBER: _ClassVar[int]
    START_LAYER_INDEX_FIELD_NUMBER: _ClassVar[int]
    total_rate: float
    start_layer_index: int
    def __init__(self, total_rate: _Optional[float] = ..., start_layer_index: _Optional[int] = ...) -> None: ...

class LoadRequest(_message.Message):
    __slots__ = ("start", "end")
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    start: int
    end: int
    def __init__(self, start: _Optional[int] = ..., end: _Optional[int] = ...) -> None: ...

class LoadResponse(_message.Message):
    __slots__ = ("layers_loaded",)
    LAYERS_LOADED_FIELD_NUMBER: _ClassVar[int]
    layers_loaded: bool
    def __init__(self, layers_loaded: bool = ...) -> None: ...
