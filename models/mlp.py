import torch
import torch.nn as nn

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, mlp_bias: bool):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=mlp_bias)

    def forward(self, input_tensor: torch.Tensor):
        return self.down_proj(nn.functional.silu(self.gate_proj(input_tensor)) * self.up_proj(input_tensor))