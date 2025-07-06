import torch.nn as nn
from ..v1_graph_ode import V1_GraphODE

class StaticNodeCore(nn.Module):
    """
    A core that uses the v1 Graph-based Neural ODE, but its weights
    are frozen after initialization.

    This tests whether "continuous depth" alone (without fluid rewiring or
    training the dynamics function) can explain performance.
    """
    def __init__(self, vibe_dim=64, num_agents=4):
        super().__init__()
        # The ODE operates on a different shape: [B, NUM_AGENTS, VIBE_DIM]
        # So we need to reshape the input to the core.
        self.vibe_dim = vibe_dim
        self.num_agents = num_agents
        self.d_model = vibe_dim * num_agents # This must match the host's d_model (256)

        self.core = V1_GraphODE(
            vibe_dim=self.vibe_dim,
            num_agents=self.num_agents
        )

        # Freeze the weights
        for param in self.core.parameters():
            param.requires_grad = False
        
        print("--- StaticNodeCore: V1 GraphODE weights have been frozen. ---")

    def forward(self, x):
        # x shape: [B, T, D]
        # We only care about the first token as the representation to evolve.
        x_first_token = x[:, 0, :]
        
        # Reshape for the ODE core: [B, D] -> [B, NUM_AGENTS, VIBE_DIM]
        batch_size = x_first_token.shape[0]
        ode_input = x_first_token.view(batch_size, self.num_agents, self.vibe_dim)

        # Run the ODE. It returns a single tensor for the final time step.
        ode_output = self.core(ode_input) # Shape: [B, NUM_AGENTS, VIBE_DIM]
        
        # Flatten and expand back to sequence length for the host network
        final_flat = ode_output.view(batch_size, -1)
        
        # We need to return a sequence of shape [B, T, D]
        # We will just broadcast the result across the time dimension
        return final_flat.unsqueeze(1).expand(-1, x.shape[1], -1) 