"""
cql.py — Conservative Q-Learning (CQL) for offline RL.

Paper: "Conservative Q-Learning for Offline Reinforcement Learning"
       (Kumar et al., NeurIPS 2020) https://arxiv.org/abs/2006.04779

Theory exposed:
  - Core insight: standard off-policy Q-learning with a fixed dataset
    produces Q-values that are *systematically overestimated* on OOD
    state-action pairs. The agent then exploits these overestimates,
    selecting actions that look good to the critic but are never actually
    executed — causing poor real-world performance.

  - CQL's solution: augment the standard Bellman loss with a penalty that
    *lower-bounds* the true Q-function under the learned policy. The key
    theoretical result (Theorem 3.2 in the paper):
        Q_CQL(s,a) <= Q^pi(s,a)  for all (s,a)
    i.e., CQL is *conservative* — it provably underestimates Q-values,
    so policy improvement on these conservative estimates is safe.

  - CQL loss (simplified form):
        L_CQL = L_Bellman  +  alpha * E_{s~D}[log sum_a exp(Q(s,a)) - E_{a~mu}[Q(s,a)]]
      The first term (logsumexp over actions) pushes Q DOWN on all actions
      (sampled uniformly or from the current policy).
      The second term (E under dataset actions) pushes Q UP on data actions.
      Net effect: Q is high on dataset actions, low on unobserved actions.

  - Automatic alpha tuning (Lagrangian formulation):
        min_alpha  alpha * (E[log sum exp Q - E_data[Q]] - target_entropy)
    This treats alpha as a Lagrange multiplier with a constraint that
    the conservative penalty doesn't over-penalize. Avoids manual tuning.

  - Logsumexp trick (numerical stability):
        log sum_k exp(Q_k) = max(Q) + log sum_k exp(Q_k - max(Q))
    Prevents overflow when Q-values are large. PyTorch's torch.logsumexp
    handles this automatically.

  - Uniform action sampling for logsumexp:
    We approximate E_{a~Uniform}[Q(s,a)] by sampling N random actions
    per state. This Monte Carlo estimate is unbiased under the uniform
    distribution and cheap to compute.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from copy import deepcopy

from src.utils.networks import TwinCritic, SquashedGaussianActor
from src.data.replay_buffer import Batch


# Number of random actions sampled per state for the logsumexp estimate
_N_RANDOM_ACTIONS = 10


class CQL:
    """
    Conservative Q-Learning for offline reinforcement learning.

    Implements the SAC-style CQL variant with:
      - Clipped double Q (twin critics)
      - Squashed Gaussian actor with entropy regularization
      - CQL penalty: logsumexp over random + policy actions minus data actions
      - Optional automatic temperature (alpha) tuning via Lagrangian

    Args:
        state_dim:           obs dimension
        action_dim:          action dimension
        max_action:          action magnitude bound
        discount:            reward discount gamma
        tau:                 Polyak update rate for target networks
        actor_lr:            actor learning rate
        critic_lr:           critic learning rate
        temp_lr:             temperature (SAC entropy) learning rate
        cql_alpha:           CQL conservative penalty weight
        cql_tau:             target entropy threshold for auto-alpha (None = fixed alpha)
        target_entropy:      SAC target entropy (default: -action_dim)
        n_random:            number of random actions for logsumexp estimate
        with_lagrange:       use Lagrangian formulation for auto cql_alpha tuning
        lagrange_threshold:  target for CQL gap in Lagrangian formulation
        device:              'cpu' or 'cuda'
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        discount: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        cql_alpha: float = 1.0,
        target_entropy: Optional[float] = None,
        n_random: int = _N_RANDOM_ACTIONS,
        with_lagrange: bool = False,
        lagrange_threshold: float = 5.0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.n_random = n_random
        self.with_lagrange = with_lagrange
        self.lagrange_threshold = lagrange_threshold

        # ---- Networks ----
        self.actor = SquashedGaussianActor(state_dim, action_dim, max_action=max_action).to(self.device)
        self.critic = TwinCritic(state_dim, action_dim, use_layer_norm=True).to(self.device)
        self.critic_target = deepcopy(self.critic)

        # ---- Optimizers ----
        self.actor_optim  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ---- SAC entropy temperature (log alpha for numerical stability) ----
        self.target_entropy = target_entropy if target_entropy else -float(action_dim)
        self.log_temp = torch.zeros(1, requires_grad=True, device=self.device)
        self.temp_optim = torch.optim.Adam([self.log_temp], lr=temp_lr)

        # ---- CQL alpha (conservative penalty weight) ----
        if with_lagrange:
            # Lagrangian: treat cql_alpha as a learnable multiplier
            self.log_cql_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.cql_alpha_optim = torch.optim.Adam([self.log_cql_alpha], lr=temp_lr)
        else:
            self.cql_alpha = cql_alpha

        self._total_steps = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def temperature(self) -> float:
        """SAC entropy temperature (positive scalar)."""
        return self.log_temp.exp().item()

    @property
    def _cql_alpha_val(self) -> torch.Tensor:
        if self.with_lagrange:
            return self.log_cql_alpha.exp().clamp(min=0.0)
        return torch.tensor(self.cql_alpha, device=self.device)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, batch: Batch) -> Dict[str, float]:
        """Full CQL update step."""
        self._total_steps += 1
        metrics = {}

        # Critic: Bellman + CQL penalty
        metrics.update(self._update_critic(batch))

        # Actor: SAC-style entropy-regularized policy
        metrics.update(self._update_actor(batch))

        # SAC temperature auto-tuning
        metrics.update(self._update_temperature(batch))

        # CQL alpha auto-tuning (if Lagrangian)
        if self.with_lagrange:
            metrics.update(self._update_cql_alpha(metrics.get("cql_gap", 0.0)))

        # Target networks
        self._soft_update(self.critic, self.critic_target)

        return metrics

    # ------------------------------------------------------------------
    # Critic Update: Bellman + CQL Conservative Penalty
    # ------------------------------------------------------------------

    def _update_critic(self, batch: Batch) -> Dict[str, float]:
        """
        CQL critic loss = standard Bellman MSE + alpha * CQL_penalty

        CQL penalty (per critic):
            penalty = logsumexp_Q - E_data[Q]

        where logsumexp_Q estimates log E_{a~pi+unif}[exp Q(s,a)]:
            ≈ log mean exp(Q(s, a_pi)) + log mean exp(Q(s, a_random))

        This is the 'H-CQL' variant combining policy actions + random actions.
        Using only policy actions ('soft CQL') is also valid but less stable.
        """
        B = batch.observations.shape[0]
        S = batch.observations  # (B, state_dim)

        # ---- Bellman target ----
        with torch.no_grad():
            next_a, next_log_pi = self.actor(batch.next_observations)
            q1_next, q2_next = self.critic_target(batch.next_observations, next_a)
            # SAC target includes entropy bonus: r + gamma*(Q - temp*log_pi)
            q_next = torch.min(q1_next, q2_next) - self.log_temp.exp() * next_log_pi
            bellman_target = batch.rewards + self.discount * (1.0 - batch.dones) * q_next

        # Current Q on dataset actions
        q1_data, q2_data = self.critic(S, batch.actions)  # (B, 1) each

        bellman_loss = F.mse_loss(q1_data, bellman_target) + F.mse_loss(q2_data, bellman_target)

        # ---- CQL penalty ----
        # 1. Sample actions from current policy (B x 1 samples)
        with torch.no_grad():
            pi_actions, pi_log_probs = self.actor(
                S.unsqueeze(1).expand(-1, self.n_random, -1).reshape(-1, S.shape[-1])
            )
            # pi_actions: (B*n_random, act_dim)
            pi_actions   = pi_actions.view(B, self.n_random, -1)
            pi_log_probs = pi_log_probs.view(B, self.n_random, 1)

        # 2. Sample uniformly random actions
        rand_actions = torch.FloatTensor(B, self.n_random, self.action_dim).uniform_(
            -self.max_action, self.max_action
        ).to(self.device)
        # Log prob of uniform over [-max_a, max_a]: constant = -act_dim * log(2*max_a)
        rand_log_probs = torch.full(
            (B, self.n_random, 1),
            fill_value=-self.action_dim * np.log(2 * self.max_action),
            device=self.device,
        )

        # 3. Evaluate Q on all sampled actions
        def _q_values(critic_fn, obs, actions):
            """Evaluate Q for (B, N, act_dim) actions. Returns (B, N, 1).
            critic_fn is a nn.Sequential that expects cat([obs, act]) as input."""
            N = actions.shape[1]
            obs_exp  = obs.unsqueeze(1).expand(-1, N, -1).reshape(-1, obs.shape[-1])
            act_flat = actions.reshape(-1, actions.shape[-1])
            sa = torch.cat([obs_exp, act_flat], dim=-1)  # (B*N, obs+act)
            q = critic_fn(sa)                            # (B*N, 1)
            return q.view(B, N, 1)

        q1_pi   = _q_values(self.critic.q1, S, pi_actions)    # (B, N, 1)
        q2_pi   = _q_values(self.critic.q2, S, pi_actions)
        q1_rand = _q_values(self.critic.q1, S, rand_actions)  # (B, N, 1)
        q2_rand = _q_values(self.critic.q2, S, rand_actions)

        # 4. logsumexp over (policy + random) actions, importance-weighted
        # Concatenate along action dimension → (B, 2N, 1)
        q1_cat = torch.cat([q1_pi - pi_log_probs, q1_rand - rand_log_probs], dim=1)
        q2_cat = torch.cat([q2_pi - pi_log_probs, q2_rand - rand_log_probs], dim=1)

        # logsumexp - log(2N): normalizes for number of samples
        # Shape: (B, 1) after logsumexp over dim=1
        cql1_logsumexp = torch.logsumexp(q1_cat, dim=1) - np.log(2 * self.n_random)
        cql2_logsumexp = torch.logsumexp(q2_cat, dim=1) - np.log(2 * self.n_random)

        # CQL penalty: logsumexp - E_data[Q]
        # Positive when Q(OOD) > Q(data), which is the pathological case
        cql_penalty_1 = (cql1_logsumexp - q1_data).mean()
        cql_penalty_2 = (cql2_logsumexp - q2_data).mean()
        cql_penalty = cql_penalty_1 + cql_penalty_2

        # Total critic loss
        critic_loss = bellman_loss + self._cql_alpha_val * cql_penalty

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()

        return {
            "critic_loss":    critic_loss.item(),
            "bellman_loss":   bellman_loss.item(),
            "cql_penalty":    cql_penalty.item(),
            "q1_data_mean":   q1_data.mean().item(),
            "q1_rand_mean":   q1_rand.mean().item(),
            # Q gap: how much Q(OOD) > Q(data) before penalty
            "cql_gap":        (cql1_logsumexp - q1_data).mean().item(),
        }

    # ------------------------------------------------------------------
    # Actor Update: SAC-style entropy-regularized
    # ------------------------------------------------------------------

    def _update_actor(self, batch: Batch) -> Dict[str, float]:
        """
        SAC actor loss: maximize E[Q(s,a)] - temperature * E[log pi(a|s)]

        The entropy term E[-log pi(a|s)] encourages exploration within the
        support of the dataset. In offline RL this also helps prevent the
        policy from collapsing to a single deterministic action too early.

        Note: we *freeze* the critic during actor update (no_grad on critic)
        to prevent actor gradients from affecting critic weights.
        """
        a, log_prob = self.actor(batch.observations)
        q1, q2 = self.critic(batch.observations, a)
        q_min = torch.min(q1, q2)

        actor_loss = (self.log_temp.exp().detach() * log_prob - q_min).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optim.step()

        return {
            "actor_loss":   actor_loss.item(),
            "log_prob_mean": log_prob.mean().item(),
            "entropy":       -log_prob.mean().item(),
        }

    # ------------------------------------------------------------------
    # SAC Temperature Update
    # ------------------------------------------------------------------

    def _update_temperature(self, batch: Batch) -> Dict[str, float]:
        """
        Automatic SAC entropy temperature tuning.

        We solve: min_alpha  alpha * (-log_pi - target_entropy)
        The target_entropy is a hyperparameter that controls how 'exploratory'
        the policy should be. A common default is -action_dim (one bit of
        entropy per action dimension).

        When actual entropy < target: alpha increases → entropy bonus grows
        When actual entropy > target: alpha decreases → policy sharpens
        """
        with torch.no_grad():
            _, log_prob = self.actor(batch.observations)

        temp_loss = -(self.log_temp * (log_prob + self.target_entropy)).mean()

        self.temp_optim.zero_grad()
        temp_loss.backward()
        self.temp_optim.step()

        return {
            "temperature": self.temperature,
            "temp_loss":   temp_loss.item(),
        }

    # ------------------------------------------------------------------
    # CQL Alpha Auto-tuning (Lagrangian)
    # ------------------------------------------------------------------

    def _update_cql_alpha(self, cql_gap: float) -> Dict[str, float]:
        """
        Lagrangian update for CQL alpha.

        We treat alpha as a Lagrange multiplier with constraint:
            cql_gap <= lagrange_threshold
        where cql_gap = E[logsumexp Q - Q_data].

        When gap > threshold: alpha increases → stronger conservative penalty
        When gap < threshold: alpha decreases → less aggressive constraint
        """
        cql_gap_t = torch.tensor(cql_gap, device=self.device)
        alpha_loss = self.log_cql_alpha * (cql_gap_t - self.lagrange_threshold)

        self.cql_alpha_optim.zero_grad()
        alpha_loss.backward()
        self.cql_alpha_optim.step()

        return {"cql_alpha": self._cql_alpha_val.item()}

    # ------------------------------------------------------------------
    # Target network
    # ------------------------------------------------------------------

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        for param, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * param.data + (1 - self.tau) * tp.data)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor.get_action(obs_t, deterministic=deterministic)
        return action.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_temp":      self.log_temp,
            "total_steps":   self._total_steps,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_temp = ckpt["log_temp"]
        self._total_steps = ckpt["total_steps"]