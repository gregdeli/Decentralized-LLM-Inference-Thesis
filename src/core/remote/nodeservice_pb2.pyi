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
    __slots__ = ("tensor_data", "tensor_shape", "dtype", "max_returned_tokens", "seq_length", "input_pos")
    TENSOR_DATA_FIELD_NUMBER: _ClassVar[int]
    TENSOR_SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    MAX_RETURNED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    SEQ_LENGTH_FIELD_NUMBER: _ClassVar[int]
    INPUT_POS_FIELD_NUMBER: _ClassVar[int]
    tensor_data: bytes
    tensor_shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    max_returned_tokens: int
    seq_length: int
    input_pos: int
    def __init__(self, tensor_data: _Optional[bytes] = ..., tensor_shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., max_returned_tokens: _Optional[int] = ..., seq_length: _Optional[int] = ..., input_pos: _Optional[int] = ...) -> None: ...

class InferenceResponse(_message.Message):
    __slots__ = ("tensor_data", "tensor_shape", "dtype")
    TENSOR_DATA_FIELD_NUMBER: _ClassVar[int]
    TENSOR_SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    tensor_data: bytes
    tensor_shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    def __init__(self, tensor_data: _Optional[bytes] = ..., tensor_shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ...) -> None: ...
