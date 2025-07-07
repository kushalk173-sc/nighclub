import torch
import torch.nn as nn
import math
from utils.dev import get_device

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class StaticTransformerXLCore(nn.Module):
    """
    A custom transformer implementation inspired by Transformer-XL.
    """
    def __init__(self, d_model=256, n_layer=6, n_head=8, d_inner=1024):
        super().__init__()
        device = get_device()
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model).to(device)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_inner,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer).to(device)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model).to(device)

    def forward(self, x, **kwargs):
        # x shape: [B, T, D]
        # Add positional encoding
        x = x + self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Final layer norm
        x = self.norm(x)
        
        return x 