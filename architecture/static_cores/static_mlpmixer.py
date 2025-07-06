import torch
import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MixerBlock(nn.Module):
    """
    An MLP-Mixer block with token-mixing and channel-mixing MLPs.
    """
    def __init__(self, dim, num_tokens, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Mlp(in_features=num_tokens, hidden_features=tokens_mlp_dim),
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Mlp(in_features=dim, hidden_features=channels_mlp_dim),
        )

    def forward(self, x):
        # x shape: [B, T, D]
        x = x + self.token_mix(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mix(x)
        return x

class StaticMlpMixerCore(nn.Module):
    """
    A stack of MLP-Mixer blocks.
    Tests if attention is critical, or if simple token/channel mixing is enough.
    """
    def __init__(self, d_model=256, depth=6, num_tokens=197, tokens_mlp_dim=2048, channels_mlp_dim=1024):
        super().__init__()
        # Note: num_tokens is a fixed parameter here, which might be brittle
        # for variable-length sequences from text/audio. Using a default
        # based on vision (196 patches + 1 CLS).
        self.blocks = nn.ModuleList([
            MixerBlock(d_model, num_tokens, tokens_mlp_dim, channels_mlp_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x) 