import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # Q,K,V: (B, heads, T, d_k)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, heads, T, T)
        if mask is not None:
            # mask broadcastable to scores
            scores = scores.masked_fill(mask == 0, float('-1e9'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, heads, T, d_k)
        return out, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # x: (B, T, d_model) -> (B, heads, T, d_k)
        B, T, _ = x.size()
        x = x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        return x

    def combine_heads(self, x):
        # x: (B, heads, T, d_k) -> (B, T, d_model)
        B, H, T, d_k = x.size()
        x = x.transpose(1, 2).contiguous().view(B, T, H * d_k)
        return x

    def forward(self, x, mask=None):
        # x: (B, T, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        out, attn = self.attention(Q, K, V, mask=mask)
        out = self.combine_heads(out)
        out = self.W_o(out)
        out = self.dropout(out)
        return out, attn
