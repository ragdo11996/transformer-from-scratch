import torch
import torch.nn as nn
from attention import MultiHeadAttention
from ffn import PositionwiseFFN

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x: (B, T, d_model)
        attn_out, attn = self.mha(x, mask=mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x, attn

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512, n_layers=2, max_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.d_model = d_model

    def forward(self, idx, mask=None):
        # idx: (B, T)
        x = self.token_emb(idx) * (self.d_model ** 0.5)
        x = self.pos_emb(x)
        attns = []
        for layer in self.layers:
            x, attn = layer(x, mask=mask)
            attns.append(attn)
        logits = self.lm_head(x)
        return logits, attns

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, d_model)
        T = x.size(1)
        x = x + self.pe[:, :T, :].to(x.device)
        return x
