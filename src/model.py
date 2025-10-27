# wrapper model (thin)
from encoder import TransformerEncoder
import torch.nn as nn

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512, n_layers=2, max_len=512, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, n_heads, d_ff, n_layers, max_len, dropout)

    def forward(self, idx, mask=None):
        return self.encoder(idx, mask=mask)
