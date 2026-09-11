# models/text_encoder.py
import math, torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, nhead=8, num_layers=6, dim_ff=2048, max_len=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos = PositionalEncoding(embed_dim, max_len)
        layer = nn.TransformerEncoderLayer(embed_dim, nhead, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_ids, attn_mask=None):
        x = self.embed(token_ids) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=~attn_mask if attn_mask is not None else None)
        
        if attn_mask is not None:
            mask = attn_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1e-9))
        else:
            x = x.mean(dim=1)

        x = self.proj(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x

