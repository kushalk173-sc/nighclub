import torch.nn as nn
from perceiver_pytorch import Perceiver

class PerceiverIOCore(nn.Module):
    """
    A Perceiver-IO model.
    This architecture uses cross-attention to distill information from a large
    input sequence into a smaller set of latent vectors, which are then processed.
    """
    def __init__(self, d_model=256, depth=6, num_latents=256, latent_dim=256, nhead=8):
        super().__init__()
        self.core = Perceiver(
            num_freq_bands=64,
            depth=depth,
            max_freq=10.,
            input_channels=d_model,
            num_latents=num_latents,
            latent_dim=latent_dim,
            num_classes=d_model, # Output is a sequence of this dimension
            final_classifier_head=False # We want the sequence output
        )

    def forward(self, x, **kwargs):
        # Perceiver expects a 'mask' for variable length sequences,
        # but for this baseline we assume fixed-length inputs from the host.
        # The model returns shape [B, T, D_MODEL], which is what we need.
        return self.core(x) 