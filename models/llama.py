"""
- What layers do I need.
- How do I tokenizer it?
- How do i invoke it with a request?
"""

"""
Layers

- input layer-norm

- attention block
    - q, k, v
    - self-attention
    - output projection

layer-norm

residual
    
- mlp
    - up proj
    - down proj

residual?
"""

import json
from typing import Any

import math
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.rms_norm = nn.RMSNorm(hidden_size, eps)

    def forward(self, input_tensor: torch.Tensor):
        return self.rms_norm(input_tensor)


#   "rope_scaling": {
#     "factor": 32.0,
#     "high_freq_factor": 4.0,
#     "low_freq_factor": 1.0,
#     "original_max_position_embeddings": 8192,
#     "rope_type": "llama3"
#   },

# todo - set default device to gpu / cpu after check so you don't need to do .to(device) everywhere

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()

    
    def forward(self, x):
        pass

class MLP(nn.Module):
    def __init__(self, config: dict[str, torch.Tensor]):
        super().__init__()

    def forward(x, input_tensor: torch.Tensor):
        pass
class DecoderLayer(nn.Module):
    """
    This is one block of the Llama GPT architecture.
    """
    def __init__(self, config: dict[str, torch.Tensor]):
        super().__init__()
        
    def forward(self, input_tensor: torch.Tensor):
        pass

class LlamaModel:
    def __init__(self, config: dict[str, torch.Tensor]):
        self.embedding = RotaryEmbedding(config)
        # todo - i think this is supposed to be torch.stack or something.
        self.layers = [DecoderLayer(config) for n in config.get("num_hidden_layers", 0)]
    
    def forward(self, x : torch.Tensor):
        pass

class LlamaForCausalLM:
    def __init__(self, config: dict[str, torch.Tensor]):
        self.model = LlamaModel(config)
        self.lm_head = None # TODO
    
    def forward(self, x : torch.Tensor):
        pass

