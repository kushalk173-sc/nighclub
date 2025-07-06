import torch
import torch.nn as nn
from torchdiffeq import odeint

class AffinityDynamics(nn.Module):
    """
    The function that defines the continuous dynamics of the "vibe".
    v3: This is f(h, t) implemented with self-attention that is biased
    by learned role embeddings (tau), representing innate "affinity".
    This class follows the user's v3 technical blueprint.
    """
    def __init__(self, num_neurons, hidden_dim, role_dim):
        super().__init__()
        # Learnable projections for Q, K, V
        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # The key v3 component: learnable, persistent role embeddings
        self.tau = nn.Parameter(torch.randn(num_neurons, role_dim))

    def forward(self, t, h):
        # h has shape [num_neurons, hidden_dim]

        # Step A: Project state for attention
        q = self.w_q(h)
        k = self.w_k(h)
        v = self.w_v(h)

        # Step B: Calculate state-based scores
        scores_state = torch.matmul(q, k.transpose(-1, -2))

        # Step C: Calculate affinity scores from roles
        scores_affinity = torch.matmul(self.tau, self.tau.transpose(-1, -2))

        # Step D: Combine scores and compute attention weights
        scores_final = scores_state + scores_affinity
        scale = h.size(-1) ** 0.5
        attention_weights = torch.softmax(scores_final / scale, dim=-1)

        # Step E: Calculate the state update (dh/dt)
        dh_dt = torch.matmul(attention_weights, v)

        return dh_dt

class V3_AttentionODE(nn.Module):
    """
    The main ODE model for the v3 architecture. It takes an initial "vibe" and
    evolves it over a continuous time period according to the AffinityDynamics.
    """
    def __init__(self, vibe_dim, role_dim, num_agents, time_steps=4):
        super().__init__()
        self.ode_net = AffinityDynamics(num_neurons=num_agents, hidden_dim=vibe_dim, role_dim=role_dim)
        self.integration_time = torch.linspace(0., 1., time_steps)

    def forward(self, initial_vibe):
        # The initial_vibe is h(0)
        self.integration_time = self.integration_time.to(initial_vibe.device)
        # Using a robust solver for stability
        all_vibes = odeint(self.ode_net, initial_vibe, self.integration_time, method='rk4', options=dict(step_size=0.25))
        
        # We return the final "vibe", h(T)
        final_vibe = all_vibes[-1]
        return final_vibe 