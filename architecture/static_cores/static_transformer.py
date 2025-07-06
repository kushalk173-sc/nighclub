import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class StaticTransformerCore(nn.Module):
    """
    A standard Transformer Encoder block.
    Serves as the "plain vanilla" yard-stick for static architectures.
    """
    def __init__(self, d_model=256, nhead=8, d_ff=1024, depth=6, activation="gelu"):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model, 
            nhead, 
            d_ff, 
            batch_first=True, 
            activation=activation
        )
        self.core = TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        # Input x: [B, T, D_MODEL]
        return self.core(x) 