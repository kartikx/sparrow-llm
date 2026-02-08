import math
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
    def forward(self, input_tensor: torch.Tensor):
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

        q, k = self.rotary_embedding(q, k)

        # convert [h0, h1, .., h7] -> [h0, h0, .. h0] [h1, h1 .. h1] ... [h7, h7 .. h7]
        k = k.repeat_interleave(
            self.num_attention_heads // self.num_key_value_heads, dim=1)
        v = v.repeat_interleave(
            self.num_attention_heads // self.num_key_value_heads, dim=1)

        att = self.self_attention(q, k, v)  # [B, h_q, T, D]

        att = att.transpose(1, 2).reshape(B, T, -1)
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
