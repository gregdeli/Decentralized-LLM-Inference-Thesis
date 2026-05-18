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
    __slots__ = ("tensor_data", "tensor_shape", "dtype", "max_returned_tokens", "seq_length", "input_pos", "response_address", "block_scales")
    TENSOR_DATA_FIELD_NUMBER: _ClassVar[int]
    TENSOR_SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    MAX_RETURNED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    SEQ_LENGTH_FIELD_NUMBER: _ClassVar[int]
    INPUT_POS_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    BLOCK_SCALES_FIELD_NUMBER: _ClassVar[int]
    tensor_data: bytes
    tensor_shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    max_returned_tokens: int
    seq_length: int
    input_pos: int
    response_address: str
    block_scales: bytes
    def __init__(self, tensor_data: _Optional[bytes] = ..., tensor_shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., max_returned_tokens: _Optional[int] = ..., seq_length: _Optional[int] = ..., input_pos: _Optional[int] = ..., response_address: _Optional[str] = ..., block_scales: _Optional[bytes] = ...) -> None: ...

class InferenceResponse(_message.Message):
    __slots__ = ("tensor_data", "tensor_shape", "dtype", "error_message")
    TENSOR_DATA_FIELD_NUMBER: _ClassVar[int]
    TENSOR_SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    tensor_data: bytes
    tensor_shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    error_message: str
    def __init__(self, tensor_data: _Optional[bytes] = ..., tensor_shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., error_message: _Optional[str] = ...) -> None: ...

class ReallocateRequest(_message.Message):
    __slots__ = ("total_rate", "start_layer_index", "predecessor_info")
    TOTAL_RATE_FIELD_NUMBER: _ClassVar[int]
    START_LAYER_INDEX_FIELD_NUMBER: _ClassVar[int]
    PREDECESSOR_INFO_FIELD_NUMBER: _ClassVar[int]
    total_rate: float
    start_layer_index: int
    predecessor_info: bytes
    def __init__(self, total_rate: _Optional[float] = ..., start_layer_index: _Optional[int] = ..., predecessor_info: _Optional[bytes] = ...) -> None: ...

class ReallocateResponse(_message.Message):
    __slots__ = ("success", "requires_restart", "bottleneck_node")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    REQUIRES_RESTART_FIELD_NUMBER: _ClassVar[int]
    BOTTLENECK_NODE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    requires_restart: bool
    bottleneck_node: str
    def __init__(self, success: bool = ..., requires_restart: bool = ..., bottleneck_node: _Optional[str] = ...) -> None: ...

class RemoveClientKVCacheRequest(_message.Message):
    __slots__ = ("client_id", "new_max_seq_length")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    NEW_MAX_SEQ_LENGTH_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    new_max_seq_length: int
    def __init__(self, client_id: _Optional[str] = ..., new_max_seq_length: _Optional[int] = ...) -> None: ...

class UpdateChainStatusRequest(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: str
    def __init__(self, status: _Optional[str] = ...) -> None: ...
