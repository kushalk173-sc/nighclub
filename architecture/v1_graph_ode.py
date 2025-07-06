import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torchdiffeq import odeint

# This file restores the v1 architecture: a Neural ODE with a fixed graph structure.

# The "Nightclub Layout" as a fixed Adjacency Matrix
club_layout = torch.tensor([
    [0, 1, 1, 1],  # MainFloor ->
    [1, 0, 0, 0],  # VIP_Lounge
    [1, 0, 0, 1],  # QuietBar
    [1, 0, 1, 0]   # OutdoorPatio
], dtype=torch.float)

class V1_GraphODENet(nn.Module):
    """
    The dynamics function f(h, t) for the v1 model.
    It uses a fixed graph (GCNConv) to define the "flow" between rooms.
    """
    def __init__(self, vibe_dim):
        super().__init__()
        self.graph_conv = GCNConv(vibe_dim, vibe_dim)
        self.activation = nn.Tanh()
        # Pre-calculate edge_index for efficiency
        self.edge_index = club_layout.nonzero().t().contiguous()

    def forward(self, t, h):
        # h has shape (num_rooms, vibe_dim)
        edge_index = self.edge_index.to(h.device)
        h = self.graph_conv(h, edge_index)
        return self.activation(h)

class V1_GraphODE(nn.Module):
    """ The main v1 ODE model. """
    def __init__(self, vibe_dim, num_agents, time_steps=4):
        super().__init__()
        self.ode_net = V1_GraphODENet(vibe_dim)
        self.integration_time = torch.linspace(0., 1., time_steps)

    def forward(self, initial_vibe):
        self.integration_time = self.integration_time.to(initial_vibe.device)
        all_vibes = odeint(self.ode_net, initial_vibe, self.integration_time)
        return all_vibes[-1] 