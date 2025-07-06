import torch.nn as nn

class Conv1DBlock(nn.Module):
    """
    A single block of 1D convolutions, consisting of a depthwise separable
    convolution followed by a pointwise convolution (in the form of a 1x1 conv).
    Includes LayerNorm and a GELU activation.
    """
    def __init__(self, d_model, kernel_size=3, expansion_factor=2):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model, # This makes it a depthwise convolution
            padding='same'
        )
        self.norm = nn.LayerNorm(d_model)
        self.pw_conv1 = nn.Linear(d_model, d_model * expansion_factor)
        self.activation = nn.GELU()
        self.pw_conv2 = nn.Linear(d_model * expansion_factor, d_model)

    def forward(self, x):
        # x shape: [B, T, D] -> [B, D, T] for conv
        x_res = x
        x = x.permute(0, 2, 1)
        x = self.dw_conv(x)
        x = x.permute(0, 2, 1) # [B, T, D]
        
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.activation(x)
        x = self.pw_conv2(x)
        
        return x + x_res

class StaticConv1DCore(nn.Module):
    """
    An "old-school" convolutional baseline using a stack of Conv1DBlocks.
    """
    def __init__(self, d_model=256, depth=6, kernel_size=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1DBlock(d_model, kernel_size) for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x 