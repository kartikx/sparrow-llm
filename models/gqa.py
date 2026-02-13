import math
from core.cache import CacheManager
from models.rope import RotaryEmbedding
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """
    Single head of self-attention
    """

    def __init__(self):
        super().__init__()

    # query -> (B, H, T, D)
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        T, D = query.shape[-2], query.shape[-1]
        causal_mask = torch.tril(torch.ones(T, T, device=query.device, dtype=torch.bool))

        attn_scores = (query @ key.transpose(-1, -2)) * (1.0 / math.sqrt(D)) # (B, H, T, T)
        attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf')) # fill upper diagonal

        return nn.functional.softmax(attn_scores, dim=-1) @ value  # (B, H, T, D)

class GroupedQueryAttention(nn.Module):
    def __init__(self, *, 
                num_attention_heads: int,
                num_key_value_heads: int,
                hidden_size: int,
                head_dim: int,
                attention_bias: bool,
                rope_theta: float,
                max_position_embeddings: int,
                rope_factor: float,
                rope_high_freq_factor: float,
                rope_low_freq_factor: float,
                rope_original_max_position_embedding: int):
        super().__init__()
        
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

        self.self_attention = SelfAttention()

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

        self.rotary_embedding = RotaryEmbedding(head_dim, rope_theta,
                                                max_position_embeddings, rope_factor, rope_high_freq_factor,
                                                rope_low_freq_factor, rope_original_max_position_embedding)

    # input_tensor [B, T, h_q*D]
    # todo for inferencing, should i get rid of the batch term?
    # todo cache manager could be set at __init__
    def forward(self, input_tensor: torch.Tensor, cache_manager: CacheManager, rid: int, layer_idx: int) -> (torch.Tensor, torch.Tensor) :
        B, T, _ = input_tensor.shape

        q: torch.Tensor = self.q_proj(input_tensor)  # [B, T, h_q * D]
        k: torch.Tensor = self.k_proj(input_tensor)  # [B, T, h_kv * D]
        v: torch.Tensor = self.v_proj(input_tensor)  # [B, T, h_kv * D]

        q = q.view(B, T, self.num_attention_heads,
                   self.head_dim).transpose(1, 2)  # [B, h_q, T, D]
        k = k.view(B, T, self.num_key_value_heads,
                   self.head_dim).transpose(1, 2)  # [B, h_kv, T, D]
        v = v.view(B, T, self.num_key_value_heads,
                   self.head_dim).transpose(1, 2)  # [B, h_kv, T, D]

        # past_k -> T, h_kv, D
        # with B -> B, T, h_kv, D
        past_k, past_v, past_len = cache_manager.get_kv(rid, layer_idx)
        if past_k is not None and past_v is not None:
            past_k = past_k.permute(1, 0, 2).unsqueeze(0)
            past_v = past_v.permute(1, 0, 2).unsqueeze(0)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        q, k = self.rotary_embedding(q, k, past_len)

        # convert [h0, h1, .., h7] -> [h0, h0, .. h0] [h1, h1 .. h1] ... [h7, h7 .. h7]
        repeat_times = self.num_attention_heads // self.num_key_value_heads 
        k = k.repeat_interleave(repeat_times, dim=1) # [B, h_q, T, D]
        v = v.repeat_interleave(repeat_times, dim=1)

        att = self.self_attention(q, k, v)  # [B, h_q, T, D]

        att = att.transpose(1, 2).reshape(B, T, -1)

        # un-interleave before inserting into kv_cache.
        k = k.view(B, self.num_key_value_heads, repeat_times, -1, self.head_dim)
        v = v.view(B, self.num_key_value_heads, repeat_times, -1, self.head_dim)

        k = k[:, :, 0, past_len:, :] # (1, h_kv, T, D]
        v = v[:, :, 0, past_len:, :]        

        cache_manager.store_kv(rid, layer_idx, k.squeeze(0).permute(1, 0, 2), v.squeeze(0).permute(1, 0, 2))

        return self.o_proj(att)

def test_self_attention():
    B, T, D = 2, 4, 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    q = torch.randn(B, T, D, device=device)
    k = torch.randn(B, T, D, device=device)
    v = torch.randn(B, T, D, device=device)

    att = SelfAttention().to(device)

    a = att(q, k, v)

    print(a.shape)

def test_grouped_query_attention():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B, T = 2, 16
    num_attention_heads = 8
    num_key_value_heads = 2
    head_dim = 16
    hidden_size = num_attention_heads * head_dim

    gqa = GroupedQueryAttention(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        hidden_size=hidden_size,
        head_dim=head_dim,
        attention_bias=False,
        rope_theta=10000.0,
        max_position_embeddings=2048,
        rope_factor=1.0,
        rope_high_freq_factor=4.0,
        rope_low_freq_factor=1.0,
        rope_original_max_position_embedding=2048,
    ).to(device)

    x = torch.randn(B, T, hidden_size, device=device)
    y = gqa(x)
    print(y.shape)

if __name__ == '__main__':
    test_grouped_query_attention()
