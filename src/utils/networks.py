"""
networks.py — Shared neural network building blocks.

Theory exposed:
  - Universal approximation theorem: why MLPs with nonlinearities can represent any Q-function
  - Layer normalization: stabilizes Q-value scale across diverse offline datasets
  - Orthogonal initialization: preserves gradient norms early in training (Saxe et al., 2013)
  - Clipped double Q-learning (TD3): two critics trained independently; take min to reduce
    overestimation bias introduced by the max operator in the Bellman backup
  - Squashed Gaussian policy (SAC-style): reparameterization trick enables low-variance
    gradients through stochastic policy; tanh squashing keeps actions in [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Utility: weight initialization
# ---------------------------------------------------------------------------

def orthogonal_init(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    """
    Orthogonal initialization (Saxe et al., 2013).

    Why: Random normal init leads to singular / near-singular weight matrices
    early in training, causing vanishing / exploding gradients. Orthogonal
    matrices preserve the L2 norm of activations (isometry), keeping the
    gradient signal healthy during early Bellman backup iterations.
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Tuple[int, ...] = (256, 256),
    activation: nn.Module = nn.ReLU,
    use_layer_norm: bool = False,
    output_activation: Optional[nn.Module] = None,
    init_gain: float = 1.0,
) -> nn.Sequential:
    """
    Construct a fully-connected MLP with optional LayerNorm.

    Layer normalization note: In offline RL, the Q-function is fit on a
    *fixed* dataset. Without normalization, Q-value magnitudes can grow
    unboundedly over many Bellman updates (a form of 'deadly triad').
    LayerNorm re-centers activations per sample, acting as an implicit
    regularizer on the representation.

    Args:
        input_dim:        dimension of input features
        output_dim:       dimension of output
        hidden_dims:      tuple of hidden layer widths
        activation:       nonlinearity class (instantiated per layer)
        use_layer_norm:   whether to insert LayerNorm after each hidden layer
        output_activation: optional activation after final linear layer
        init_gain:        gain for orthogonal initialization

    Returns:
        nn.Sequential module
    """
    layers = []
    in_dim = input_dim

    for h_dim in hidden_dims:
        linear = orthogonal_init(nn.Linear(in_dim, h_dim), gain=init_gain)
        layers.append(linear)
        if use_layer_norm:
            layers.append(nn.LayerNorm(h_dim))
        layers.append(activation())
        in_dim = h_dim

    # Output layer — smaller gain for stable initial predictions
    out_layer = orthogonal_init(nn.Linear(in_dim, output_dim), gain=0.01)
    layers.append(out_layer)

    if output_activation is not None:
        layers.append(output_activation())

    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Critic: Clipped Double Q-Network
# ---------------------------------------------------------------------------

class TwinCritic(nn.Module):
    """
    Two independent Q-networks (Twin Critic / Clipped Double Q).

    Theory: Thrun & Schwartz (1993) showed that the max operator in Bellman
    backups systematically *overestimates* Q-values due to Jensen's inequality:
        E[max Q] >= max E[Q]
    TD3 (Fujimoto et al., 2018) addresses this by training two separate critics
    Q1, Q2 and using min(Q1, Q2) as the regression target, providing a
    pessimistic but lower-variance Bellman target. This is especially critical
    in offline RL where overestimation on OOD actions can't be corrected by
    further environment interaction.

    Input:  (state, action) concatenated
    Output: scalar Q-value per network
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        use_layer_norm: bool = True,
    ):
        super().__init__()
        in_dim = state_dim + action_dim

        self.q1 = build_mlp(
            input_dim=in_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            use_layer_norm=use_layer_norm,
        )
        self.q2 = build_mlp(
            input_dim=in_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            use_layer_norm=use_layer_norm,
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return Q1(s,a) and Q2(s,a) independently."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return min(Q1, Q2) — the pessimistic Bellman target estimate."""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


# ---------------------------------------------------------------------------
# Value Network (used by IQL)
# ---------------------------------------------------------------------------

class ValueNetwork(nn.Module):
    """
    State-value function V(s) — used by Implicit Q-Learning (IQL).

    IQL (Kostrikov et al., 2021) avoids querying the policy on OOD actions
    entirely. Instead of max_a Q(s,a), it fits V(s) using *expectile regression*
    to approximate the in-support maximum. This network learns V(s) directly
    from (s, a) samples in the offline dataset.

    Input:  state s
    Output: scalar V(s)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.net = build_mlp(
            input_dim=state_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ---------------------------------------------------------------------------
# Deterministic Actor (used by TD3+BC and CQL)
# ---------------------------------------------------------------------------

class DeterministicActor(nn.Module):
    """
    Deterministic policy pi(s) -> a in [-1, 1]^action_dim.

    Used by TD3+BC and CQL (with deterministic variant).
    Tanh output squashes actions to the valid range without hard clipping,
    which preserves smooth gradients through the action boundary.

    Theory (policy gradient): The actor is trained to maximize Q(s, pi(s))
    via chain rule: d/d_theta Q(s, pi_theta(s)) = (dQ/da)(da/d_theta).
    This requires Q to be differentiable w.r.t. actions — satisfied here
    since both actor and critic are smooth MLPs.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        max_action: float = 1.0,
    ):
        super().__init__()
        self.max_action = max_action
        self.net = build_mlp(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            use_layer_norm=False,
            output_activation=nn.Tanh,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)


# ---------------------------------------------------------------------------
# Stochastic Actor — Squashed Gaussian (used by IQL / SAC-style CQL)
# ---------------------------------------------------------------------------

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class SquashedGaussianActor(nn.Module):
    """
    Stochastic policy: outputs mean & log_std, samples via reparameterization.

    Theory (reparameterization trick):
        a = tanh(mu + std * eps),  eps ~ N(0, I)
    This allows gradients to flow through the sampling operation since the
    randomness (eps) is separated from the parameters (mu, std). Without
    reparameterization, policy gradient estimators (REINFORCE) have high
    variance; the reparameterization gradient is much lower variance.

    Log-probability correction for tanh squashing:
        log pi(a|s) = log N(u|mu,std) - sum_i log(1 - tanh(u_i)^2)
    where u is the pre-squash action. This Jacobian correction is critical
    for entropy estimation in SAC-style objectives.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        max_action: float = 1.0,
    ):
        super().__init__()
        self.max_action = max_action
        self.action_dim = action_dim

        # Shared trunk
        self.trunk = build_mlp(
            input_dim=state_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            use_layer_norm=False,
        )
        # Separate heads for mean and log_std
        self.mu_head = orthogonal_init(nn.Linear(hidden_dims[-1], action_dim), gain=0.01)
        self.log_std_head = orthogonal_init(nn.Linear(hidden_dims[-1], action_dim), gain=0.01)

    def forward(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action:   sampled (or deterministic) squashed action
            log_prob: log pi(a|s) with tanh Jacobian correction
        """
        h = self.trunk(state)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mu, std)

        if deterministic:
            u = mu
        else:
            # Reparameterization: u = mu + std * eps
            u = dist.rsample()

        # Tanh squash
        a = torch.tanh(u)

        # Log-prob with Jacobian correction
        # log(1 - tanh(u)^2) = log(sech^2(u)), numerically stable via:
        log_prob = dist.log_prob(u) - torch.log(1.0 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return self.max_action * a, log_prob

    def get_action(self, state: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Convenience method for evaluation rollouts."""
        with torch.no_grad():
            action, _ = self.forward(state, deterministic=deterministic)
        return action