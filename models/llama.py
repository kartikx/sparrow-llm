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
class RotaryEmbedding(nn.Module):
    # todo - ensure that head_size -> per head.
    def __init__(self, head_size: int, rope_theta: float, max_position_embeddings: int,
                rope_factor: float, rope_high_freq_factor: float, rope_low_freq_factor: float,
                rope_original_max_position_embedding: int):
        super().__init__()
        inv_freq_table = 1.0 / (rope_theta ** (torch.arange(0, head_size, 2, dtype=torch.float) / head_size)) # (head_size//2)
        positions = torch.arange(0, max_position_embeddings, dtype=torch.float) # T
        freqs = torch.outer(positions, inv_freq_table) # T, head_size//2
        freqs = torch.cat((freqs, freqs), dim=-1) # T, head_size

        # cached for future [b, s] pairs
        self.register_buffer("cos_cache", freqs.cos(), persistent=False) # T, head_size -> cos0, cos1, cos0, cos1 per token position.
        self.register_buffer("sin_cache", freqs.sin(), persistent=False) # T, head_size -> sin0, sin1, sin0, sin1 per token position.
            
    # x -> [B, T, num_heads, head_size]
    def rotate_half(self, x: torch.Tensor):
        """
        converts [q0 q1 q2 q3] -> [-q2 -q3 q0 q1]
        """
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1]//2:]
        
        return torch.cat((-x2, x1), dim=-1)


    # query -> [B, T, num_heads, head_size]
    # key -> [B, T, num_heads, head_size]
    # self.cos_cache -> [T, head_size]
    def forward(self, query: torch.Tensor):
        print(query.shape)
        print(self.cos_cache.shape)

        # todo - :t slicing
        # todo - intermediate device to gpu
        # todo - rope scaling.
        
        t, head_size = query.shape[1], query.shape[3] 
        query = query * self.cos_cache.view(1, t, 1, head_size) + self.rotate_half(query) * self.sin_cache.view(1, t, 1, head_size)
        # key = key * self.cos + self.rotate_half(key) * self.sin
        
        return query


def test_rotary_embedding():
    rotary_embedding = RotaryEmbedding(4, 1000, 4, 1, 1, 1, 1)
    
    # (B, T, num_heads, head_size)
    # todo - verify against hf impl
    x = torch.randint(1, 4, (2, 2, 4, 4))
    
    print("====Before====")
    print(x.shape)
    print(x)
    print("========")
    
    x = rotary_embedding(x)
    print("====After====")
    print(x.shape)
    print(x)

class GroupedQueryAttention:
    def __init__(self):
        pass
    
    def forward(self, x):
        pass

class MLP(nn.Module):
    def __init__(self, config: dict[str, torch.Tensor]):
        hidden_size = config.get("hidden_size")
        self.up_proj = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.gate_proj = nn.SiLU() # todo - should inplace?
        self.down_proj = nn.Linear(hidden_size * 4, hidden_size, bias=False); 
        pass

    def forward(x, input_tensor: torch.Tensor):
        pass
class DecoderLayer(nn.Module):
    """
    This is one block of the Llama GPT architecture.
    """
    def __init__(self, config: dict[str, torch.Tensor]):
        super().__init__()
        
        self.input_norm = RMSNorm(config)
        self.self_attn = GroupedQueryAttention(config)
        self.mlp = MLP(config)
        self.post_attn_norm = RMSNorm(config)
        
    def forward(self, input_tensor: torch.Tensor):
        input_tensor = input_tensor + self.self_attn(self.input_norm(input_tensor)) 
        input_tensor = input_tensor + self.mlp(self.post_attn_norm(input_tensor))
        return input_tensor

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

if __name__ == '__main__':
    test_rotary_embedding()