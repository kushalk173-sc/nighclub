import torch
import torch.nn as nn
import math
from torchdiffeq import odeint

# v2: The "Nightclub" is an open warehouse. There is no fixed layout.
# Connections are fluid and determined by attention, representing "social gravity".

class ODENet(nn.Module):
    """
    The function that defines the continuous dynamics of the "vibe".
    v2: This is f(h, t) implemented with self-attention, where influence
    is calculated on the fly based on the current "vibe" of all agents.
    """
    def __init__(self, vibe_dim, n_heads=4):
        super().__init__()
        assert vibe_dim % n_heads == 0, "Vibe dimension must be divisible by number of heads"
        self.n_heads = n_heads
        self.head_dim = vibe_dim // n_heads

        # A single linear layer to project h to Q, K, V
        self.qkv_net = nn.Linear(vibe_dim, vibe_dim * 3)
        self.out_net = nn.Linear(vibe_dim, vibe_dim)
        self.activation = nn.Tanh()

    def forward(self, t, h):
        # h has shape (num_agents, vibe_dim). num_agents is our conceptual NUM_ROOMS.
        num_agents, vibe_dim = h.shape

        qkv = self.qkv_net(h)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention: (num_agents, n_heads, head_dim)
        q = q.view(num_agents, self.n_heads, self.head_dim)
        k = k.view(num_agents, self.n_heads, self.head_dim)
        v = v.view(num_agents, self.n_heads, self.head_dim)

        # Transpose for matmul: (n_heads, num_agents, head_dim)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # Apply attention to values
        context = torch.matmul(attention_probs, v)

        # Concatenate heads and project back to original shape
        context = context.transpose(0, 1).contiguous().view(num_agents, vibe_dim)
        update = self.out_net(context)
        
        return self.activation(update)

class NightclubODE(nn.Module):
    """
    The main ODE model. It takes an initial "vibe", and evolves it
    over a continuous time period according to the ODENet dynamics.
    """
    def __init__(self, vibe_dim, time_steps=4):
        super().__init__()
        self.ode_net = ODENet(vibe_dim)
        self.integration_time = torch.linspace(0., 1., time_steps)

    def forward(self, initial_vibe):
        # The initial_vibe is h(0)
        self.integration_time = self.integration_time.to(initial_vibe.device)
        all_vibes = odeint(self.ode_net, initial_vibe, self.integration_time)
        
        # We return the final "vibe" of the club, h(T)
        final_vibe = all_vibes[-1]
        return final_vibe 