"""Main server application logic"""

import torch
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple

from core.llm_loader import LLM


class Server:
    def __init__(
        self,
        model_path: Path,
        num_layers: int = None,
        layers_start_idx: int = 0,
        is_head: bool = False,
        successor: "Server" = None,
        time_it: bool = False,
    ) -> None:
        self.llm = LLM.load(model_path, is_client=False, num_layers=num_layers, layers_start_idx=layers_start_idx, time_it=time_it)
        self.model = self.llm.model
        self.is_head = is_head

        # Determine is the server is the tail of the chain, meaning it holds the final transformer layer.
        self.is_tail = False
        if self.llm.layers_loaded[1] == (self.llm.config["num_hidden_layers"] - 1):
            self.is_tail = True

        self.successor = successor

    @torch.no_grad()
    def run_layers(
        self,
        input: torch.Tensor,
        max_returned_tokens: int,
        seq_length: int = None,
        input_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        This function runs inference on the server's assigned transformer layers and send the output to the next node.
        """
        if not self.llm.kv_cache_initialized:
            device = self.llm.preprocessor.device
            # Na allaksw to batch_size otan kanw batched inference
            self.model.set_kv_cache(batch_size=1, max_seq_length=max_returned_tokens, device=device)
            self.llm.kv_cache_initialized = True

        h = self.model.forward_server(input, seq_length, input_pos)

        if self.is_tail:
            return h

        return self.successor.run_layers(h, max_returned_tokens, seq_length, input_pos)
