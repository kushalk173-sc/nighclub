import torch
import torch.nn as nn

class FNetBlock(nn.Module):
    """
    An FNet block that uses Fourier Transforms for token mixing.
    """
    def __init__(self, d_model, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model)
        )

    def forward(self, x):
        # x shape: [B, T, D]
        # Fourier transform mixes along the sequence (T) dimension
        x_res = x
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        x = self.norm1(x + x_res)
        
        x_res = x
        x = self.ff(x)
        x = self.norm2(x + x_res)
        
        return x

class StaticFNetCore(nn.Module):
    """
    A stack of FNet blocks.
    An attention-free alternative that keeps global mixing via FFT.
    """
    def __init__(self, d_model=256, depth=6, ff_mult=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            FNetBlock(d_model, ff_mult) for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x 