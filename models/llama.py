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
from models.gqa import GroupedQueryAttention
from models.mlp import SwiGLUFFN
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

        num_attention_heads = config.get("num_attention_heads")
        head_dim = config.get("head_dim") or (hidden_size // num_attention_heads)

        self.self_attn = GroupedQueryAttention(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=config["num_key_value_heads"],
            hidden_size=hidden_size,
            head_dim=head_dim,
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
        self.tie_word_embeddings = bool(config.get("tie_word_embeddings"))

        self.lm_head = nn.Linear(config.get("hidden_size"), 
                                 config.get("vocab_size"),
                                 bias = False)

    def tie_weights(self, expected_missing: set | None):
        self.lm_head.weight = self.model.embed_tokens.weight
        if expected_missing is not None:
            expected_missing.add("lm_head.weight")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor :
        x = self.model(input_ids)

        if x.dtype != self.lm_head.weight.dtype:
            x = x.to(self.lm_head.weight.dtype)
        return self.lm_head(x)

    def load_weights(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        strict: bool = False,
        verbose: bool = True,
    ):
        try:
            # Fast path, but need to be careful about dtypes.
            incompatible = self.load_state_dict(
                state_dict,
                strict=strict,
                assign=True,
            )
        except TypeError:
            incompatible = self.load_state_dict(state_dict, strict=strict)

        expected_missing = set()
        if self.tie_word_embeddings:
            # assign=True can replace Parameter objects; re-tie after loading.
            self.tie_weights(expected_missing)

        unexpected_missing_keys = [k for k in incompatible.missing_keys if k not in expected_missing]
        
        if verbose and (unexpected_missing_keys or incompatible.unexpected_keys):
            print("missing: ", unexpected_missing_keys)
            print("unexpected: ", incompatible.unexpected_keys)

        return incompatible
