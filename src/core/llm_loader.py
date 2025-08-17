from typing import Dict, Any, Union, Literal
from pathlib import Path
from model import Llama3


class LLM:
    def __init__(
        self,
        model: Llama3,
        preprocessor=None,
        config: Dict[str, Any] = None,
        checkpoint_dir: Path = None,
        kv_cache_initialized: bool = False,
        fixed_kv_cache_size: Union[int, Literal["max_model_supported"], None] = None,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.kv_cache_initialized = kv_cache_initialized
        self.fixed_kv_cache_size = fixed_kv_cache_size

    """High-level API for loading a Llama 3.2 model and generating text"""

    @classmethod
    def load(
        cls,
        checkpoint_dir: Path,
    ) -> "LLM":
        # Sunexizw edw
        pass
