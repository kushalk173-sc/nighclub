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

class TokenMixingMLP(nn.Module):
    """
    A token-mixing MLP that can handle variable sequence lengths.
    """
    def __init__(self, dim, tokens_mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, tokens_mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(tokens_mlp_dim, dim)
        
    def forward(self, x):
        # x shape: [B, T, D]
        # Apply token mixing: for each position, mix across the sequence
        x_norm = self.norm(x)  # [B, T, D]
        
        # Apply MLP to each position independently
        mixed = self.fc2(self.act(self.fc1(x_norm)))  # [B, T, D]
        
        return mixed

class MixerBlock(nn.Module):
    """
    An MLP-Mixer block with token-mixing and channel-mixing MLPs.
    """
    def __init__(self, dim, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.token_mix = TokenMixingMLP(dim, tokens_mlp_dim)
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Mlp(in_features=dim, hidden_features=channels_mlp_dim),
        )

    def forward(self, x):
        # x shape: [B, T, D]
        # Token mixing: apply MLP to each position
        x = x + self.token_mix(x)
        # Channel mixing: apply MLP across channels
        x = x + self.channel_mix(x)
        return x

class StaticMlpMixerCore(nn.Module):
    """
    A stack of MLP-Mixer blocks that handles variable sequence lengths.
    """
    def __init__(self, d_model=256, depth=6, tokens_mlp_dim=2048, channels_mlp_dim=1024):
        super().__init__()
        self.blocks = nn.ModuleList([
            MixerBlock(d_model, tokens_mlp_dim, channels_mlp_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: [B, T, D] where T can vary
        for block in self.blocks:
            x = block(x)
        return self.norm(x) 