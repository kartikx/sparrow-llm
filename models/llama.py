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
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from models.gqa import GroupedQueryAttention
from models.mlp import SwiGLUFFN
from models.norm import RMSNorm
from models.rope import RotaryEmbedding
from safetensors.torch import load_file
import torch

class MLP(nn.Module):
    def __init__(self, config: dict[str, torch.Tensor]):
        super().__init__()

    def forward(x, input_tensor: torch.Tensor):
        pass

# todo - how am i supposed to use the torch_dtype everywhere?


class DecoderLayer(nn.Module):
    """
    This is one block of the Llama GPT architecture.
    
    This should include Pre-LN, then GQA, then residual, then LN (?), then FFN,
    then another residual?
    """

    def __init__(self, config: dict[str, torch.Tensor]):
        super().__init__()

        hidden_size = config["hidden_size"]
        rms_norm_eps = config["rms_norm_eps"]

        rope_scaling = config["rope_scaling"]
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

        self.self_attn = GroupedQueryAttention(
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            hidden_size=hidden_size,
            head_dim=config["head_dim"],
            attention_bias=config["attention_bias"],
            rope_theta=config["rope_theta"],
            max_position_embeddings=config["max_position_embeddings"],
            rope_factor=rope_scaling["factor"],
            rope_high_freq_factor=rope_scaling["high_freq_factor"],
            rope_low_freq_factor=rope_scaling["low_freq_factor"],
            rope_original_max_position_embedding=rope_scaling["original_max_position_embeddings"],
        )

        self.mlp = SwiGLUFFN(
            hidden_size=hidden_size,
            intermediate_size=config["intermediate_size"],
            mlp_bias=config["mlp_bias"],
        )

    def forward(self, input_tensor: torch.Tensor):
        x = input_tensor
        attn_out = self.self_attn(self.input_layernorm(x))
        x = x + attn_out

        ffn_out = self.mlp(self.post_attention_layernorm(x))
        return x + ffn_out

class LlamaModel(nn.Module):
    def __init__(self, config: dict[str, torch.Tensor]):
        super().__init__()

        hidden_size = config["hidden_size"]
        rms_norm_eps = config["rms_norm_eps"]

        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.norm = nn.RMSNorm(hidden_size, rms_norm_eps)
        self.layers = nn.ModuleList([DecoderLayer(config)
                       for _ in range(config.get("num_hidden_layers", 0))])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(dtype=torch.long)

        x = self.embed_tokens(input_ids)

        for decoder in self.layers:
            x = decoder(x) 

        return self.norm(x)

class LlamaForCausalLM(nn.Module):
    def __init__(self, config: dict[str, torch.Tensor]):
        super().__init__()
        
        self.model = LlamaModel(config)

        self.lm_head = nn.Linear(config.get("hidden_size"), 
                                 config.get("vocab_size"),
                                 bias = False)
        if config.get("tie_word_embeddings"):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor :
        x = self.model(input_ids)

        return self.lm_head(x)

def load_weights(model: LlamaForCausalLM, shard_paths: list[str]):
    state_dict = {}
    for path in shard_paths:
        state_dict.update(load_file(path, device="cpu"))

    # print(state_dict.keys())

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    # print("missing: ", missing)
    # print("unexpected: ", unexpected)

if __name__ == '__main__':
    shard_paths = [Path('/scratch/bcjw/kramesh/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/model.safetensors')]
    
    with open('config.json', 'r') as f:
        config = json.loads(f.read())
        
    model = LlamaForCausalLM(config)
        
    load_weights(model, shard_paths)

    