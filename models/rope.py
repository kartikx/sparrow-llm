import math
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        max_position_embeddings: int,
        rope_factor: float,
        rope_high_freq_factor: float,
        rope_low_freq_factor: float,
        rope_original_max_position_embedding: int,
    ):
        super().__init__()
        inv_freq_table = self._compute_rope_scaling(
            head_dim,
            rope_theta,
            rope_factor,
            rope_low_freq_factor,
            rope_high_freq_factor,
            rope_original_max_position_embedding,
        )
        positions = torch.arange(
            0, max_position_embeddings, dtype=torch.float)  # T
        freqs = torch.outer(positions, inv_freq_table)  # T, head_dim//2
        freqs = torch.cat((freqs, freqs), dim=-1)  # T, head_dim

        # Cached for future [B, H, T, D] pairs. Buffers follow the module's
        # device when you call .to(device) on a parent model.
        self.register_buffer(
            "cos_cache",
            freqs.cos().unsqueeze(0).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "sin_cache",
            freqs.sin().unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    def _compute_rope_scaling(self,
                              head_dim, rope_theta, factor, low_freq_factor, high_freq_factor, original_max_position_embeddings):
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim,
                          2, dtype=torch.float) / head_dim))  # (head_size//2)

        wavelen = 2 * math.pi / inv_freq
        low_freq_wavelen = original_max_position_embeddings / low_freq_factor
        high_freq_wavelen = original_max_position_embeddings / high_freq_factor

        # wavelen < high_freq_wavelen: do nothing
        # wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (original_max_position_embeddings / wavelen -
                         low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / \
            factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * \
            ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(
            is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        return inv_freq_llama

    def rotate_half(self, x: torch.Tensor):
        """
        converts [q0 q1 q2 q3] -> [-q2 -q3 q0 q1]
        """
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1]//2:]

        return torch.cat((-x2, x1), dim=-1)

    def forward(self, query: torch.Tensor, key: torch.Tensor):
        """
        Expects Query, Key -> [B, H, T, D]
        cos_cache, sin_cache -> [1, T, D]
        """
        seq_len = query.shape[2]
        cos = self.cos_cache[..., :seq_len, :]
        sin = self.sin_cache[..., :seq_len, :]

        query = query * cos + self.rotate_half(query) * sin
        key = key * cos + self.rotate_half(key) * sin

        return query, key
