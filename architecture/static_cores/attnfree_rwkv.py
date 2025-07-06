import torch.nn as nn
from rwkv.model import RWKV as RwkvModel
from rwkv.utils import PIPELINE

class StaticRwkvCore(nn.Module):
    """
    A core using the RWKV model, a recent attention-free sequence model.
    """
    def __init__(self, d_model=256, n_layer=6, n_head=8):
        super().__init__()
        # The rwkv library requires a specific config structure.
        # We will create one on the fly.
        from types import SimpleNamespace
        args = SimpleNamespace()
        args.n_layer = n_layer
        args.n_embd = d_model
        args.vocab_size = 50277 # A default, not used for our purposes
        args.ctx_len = 4096 # Max sequence length
        
        # RWKV v4 uses these fixed values
        args.head_size_a = 64
        args.head_size_b = d_model // n_head
        args.n_head = n_head
        args.n_att = d_model // n_head
        args.n_ffn = int((d_model * 3.5) / 32) * 32
        
        self.core = RwkvModel(args)

    def forward(self, x, **kwargs):
        # RWKV model returns the output and a state for autoregressive generation.
        # We only need the output for this baseline.
        output, _ = self.core(x)
        return output 