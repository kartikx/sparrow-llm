import torch.nn as nn
from models.gqa import GroupedQueryAttention
from models.mlp import SwiGLUFFN
import torch
import logging

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, config: dict[str, torch.Tensor]):
        super().__init__()

    def forward(x, input_tensor: torch.Tensor):
        pass

class DecoderLayer(nn.Module):
    """
    This is one block of the Llama GPT architecture.
    
    This should include Pre-LN, then GQA, then residual, then LN (?), then FFN,
    then another residual?
    """

    def __init__(self, config: dict[str, torch.Tensor], layer_idx: int):
        super().__init__()

        self.layer_idx = layer_idx

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

    def forward(self, input_tensor: torch.Tensor, past_key_values: torch.Tensor):
        x = input_tensor
        attn_out, updated_key_values = self.self_attn(self.input_layernorm(x), past_key_values)

        if self.layer_idx == 0:
            logger.debug("[DecoderLayer] Past past_key_values: %s", past_key_values.shape if past_key_values is not None else None)
            logger.debug("[DecoderLayer] Updated past_key_values: %s", updated_key_values.shape)

        x = x + attn_out

        ffn_out = self.mlp(self.post_attention_layernorm(x))
        return x + ffn_out, updated_key_values

class LlamaModel(nn.Module):
    def __init__(self, config: dict[str, torch.Tensor]):
        super().__init__()

        hidden_size = config["hidden_size"]
        rms_norm_eps = config["rms_norm_eps"]

        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.norm = nn.RMSNorm(hidden_size, rms_norm_eps)
        self.layers = nn.ModuleList([DecoderLayer(config, idx)
                       for idx in range(config.get("num_hidden_layers", 0))])

    def forward(self, input_ids: torch.Tensor, past_key_values: list[torch.Tensor]) -> (torch.Tensor, list[torch.Tensor]):
        
        if past_key_values is not None and len(past_key_values) > 0:
            logger.debug("[LlamaModel] Past KV length: %d", len(past_key_values))
            logger.debug("[LlamaModel] First KV shape: %s", past_key_values[0].shape)
        
        input_ids = input_ids.to(dtype=torch.long)

        x = self.embed_tokens(input_ids)
        updated_key_values = []

        for idx, decoder in enumerate(self.layers):
            layer_past = past_key_values[idx] if len(past_key_values) > 0 and past_key_values is not None else None
            x, layer_updated = decoder(x, layer_past)
            updated_key_values.append(layer_updated)

        return self.norm(x), updated_key_values

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

    def forward(self, input_ids: torch.Tensor, past_kv_values: list[torch.Tensor]) -> (torch.Tensor, list[torch.Tensor]) :
        x, updated_kv_values = self.model(input_ids, past_kv_values)

        if x.dtype != self.lm_head.weight.dtype:
            x = x.to(self.lm_head.weight.dtype)
        return self.lm_head(x), updated_kv_values

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
            logger.warn("missing keys: %s", unexpected_missing_keys)
            logger.warn("unexpected keys: %s", incompatible.unexpected_keys)

        return incompatible
