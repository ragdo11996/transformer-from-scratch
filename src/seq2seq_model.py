import torch
import torch.nn as nn
from encoder import TransformerEncoder
from decoder import TransformerDecoder, generate_subsequent_mask

class TransformerSeq2Seq(nn.Module):
    """完整 Encoder-Decoder 模型"""
    def __init__(self, src_vocab, tgt_vocab, d_model=128, n_heads=4, d_ff=512, n_layers=2, dropout=0.1, max_len=256):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab, d_model, n_heads, d_ff, n_layers, max_len, dropout)
        self.decoder = TransformerDecoder(tgt_vocab, d_model, n_heads, d_ff, n_layers, max_len, dropout)
        self.d_model = d_model

    def forward(self, src, tgt):
        # src: (B, T_src)
        # tgt: (B, T_tgt)
        enc_out, _ = self.encoder(src)
        tgt_mask = generate_subsequent_mask(tgt.size(1)).to(tgt.device)
        logits, _, _ = self.decoder(tgt, enc_out, tgt_mask=tgt_mask)
        return logits
