import torch.nn as nn
from transformers import TransfoXLConfig, TransfoXLModel

class StaticTransformerXLCore(nn.Module):
    """
    A Transformer-XL model, which adds recurrence to the transformer architecture.
    A good foil for long-context pillars.
    """
    def __init__(self, d_model=256, n_layer=6, n_head=8, d_inner=1024):
        super().__init__()
        config = TransfoXLConfig(
            d_model=d_model,
            d_embed=d_model,
            n_layer=n_layer,
            n_head=n_head,
            d_inner=d_inner,
        )
        self.core = TransfoXLModel(config)

    def forward(self, x, **kwargs):
        # The 'mems' argument allows for statefulness across segments,
        # but for this baseline we'll process each input independently.
        return self.core(inputs_embeds=x).last_hidden_state 