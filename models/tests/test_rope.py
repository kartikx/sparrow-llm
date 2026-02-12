import unittest

import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from models.rope import RotaryEmbedding
from tests.utils import SMALL_MODEL_FOR_TEST, load_model_config


class TestRotaryEmbedding(unittest.TestCase):
    def test_matches_huggingface_output(self):
        config = load_model_config(SMALL_MODEL_FOR_TEST)

        rope_scaling = config.get("rope_scaling", {})
        head_dim = config.get("head_dim")
        num_attention_heads = config.get("num_attention_heads")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rotary_embedding = RotaryEmbedding(
            head_dim=head_dim,
            rope_theta=config.get("rope_theta"),
            max_position_embeddings=config.get("max_position_embeddings"),
            rope_factor=rope_scaling.get("factor"),
            rope_high_freq_factor=rope_scaling.get("high_freq_factor"),
            rope_low_freq_factor=rope_scaling.get("low_freq_factor"),
            rope_original_max_position_embedding=rope_scaling.get(
                "original_max_position_embeddings",
            ),
        ).to(device)

        torch.manual_seed(0)
        B, H, T, D = 2, num_attention_heads, 4, head_dim

        hf_rope = LlamaRotaryEmbedding(LlamaConfig(**config)).to(device)

        q = torch.randn(B, H, T, D, dtype=torch.float32, device=device)
        k = torch.randn(B, H, T, D, dtype=torch.float32, device=device)

        pos = torch.arange(T, device=device).unsqueeze(0).repeat(B, 1)  # [B, T]

        cos, sin = hf_rope(q, pos)

        q_hf, k_hf = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        q_sprw, k_sprw = rotary_embedding(q, k)

        self.assertEqual(q_hf.shape, q_sprw.shape,
                         msg=f"q shape mismatch: hf={q_hf.shape}, sprw={q_sprw.shape}")
        self.assertEqual(k_hf.shape, k_sprw.shape,
                         msg=f"k shape mismatch: hf={k_hf.shape}, sprw={k_sprw.shape}")

        self.assertTrue(
            torch.allclose(q_hf, q_sprw, rtol=1e-4, atol=1e-5),
            msg=f"q: max abs diff = {(q_hf - q_sprw).abs().max().item():.2e}",
        )
        self.assertTrue(
            torch.allclose(k_hf, k_sprw, rtol=1e-4, atol=1e-5),
            msg=f"k: max abs diff = {(k_hf - k_sprw).abs().max().item():.2e}",
        )
