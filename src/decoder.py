import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from ffn import PositionwiseFFN
from encoder import PositionalEncoding

class DecoderLayer(nn.Module):
    """单层 Transformer Decoder"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        """
        x: decoder input (B, T_tgt, d_model)
        enc_out: encoder output (B, T_src, d_model)
        tgt_mask: for masking future tokens (causal mask)
        memory_mask: for padding mask between src/tgt
        """
        # masked self-attention
        _x, self_attn = self.self_attn(x, mask=tgt_mask)
        x = self.norm1(x + _x)

        # encoder-decoder cross attention
        _x, cross_attn = self.cross_attn(x, mask=memory_mask)
        x = self.norm2(x + _x)

        # position-wise feed-forward
        _x = self.ffn(x)
        x = self.norm3(x + _x)

        return x, self_attn, cross_attn


class TransformerDecoder(nn.Module):
    """完整 Decoder 堆叠模块"""
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512, n_layers=2, max_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.d_model = d_model

    def forward(self, tgt_idx, enc_out, tgt_mask=None, memory_mask=None):
        x = self.token_emb(tgt_idx) * (self.d_model ** 0.5)
        x = self.pos_emb(x)
        self_attns, cross_attns = [], []
        for layer in self.layers:
            x, sa, ca = layer(x, enc_out, tgt_mask, memory_mask)
            self_attns.append(sa)
            cross_attns.append(ca)
        logits = self.lm_head(x)
        return logits, self_attns, cross_attns


def generate_subsequent_mask(sz: int):
    """生成下三角掩码，防止预测看到未来信息"""
    mask = torch.tril(torch.ones(sz, sz)).unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
    return mask
