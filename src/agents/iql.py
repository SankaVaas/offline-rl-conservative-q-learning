"""
iql.py — Implicit Q-Learning (IQL) for offline RL.

Paper: "Offline Reinforcement Learning with Implicit Q-Learning"
       (Kostrikov et al., ICLR 2022) https://arxiv.org/abs/2110.06169

Theory exposed:
  - Key motivation: both TD3+BC and CQL still query Q(s, a') for actions
    a' ~ pi that may lie outside the dataset. IQL completely avoids this by
    *never* evaluating Q on policy-generated actions during training.

  - Core trick — expectile regression to approximate max_a Q(s,a):
    The maximum of a set can be approximated without explicitly enumerating
    all elements by fitting a high quantile. Specifically, if V(s) is fit
    to approximate E_{a~mu}[Q(s,a) | Q(s,a) > V(s)] (i.e., the conditional
    expectation of Q on its upper tail), then for tau→1, V(s) → max_a Q(s,a).

  - Expectile loss: asymmetric L2 loss controlled by tau ∈ (0.5, 1):
        L_tau(u) = |tau - I(u < 0)| * u^2
    where u = Q(s,a) - V(s).
    When u > 0 (Q > V): weighted by tau    (large weight → V chases Q up)
    When u < 0 (Q < V): weighted by 1-tau  (small weight → V resists Q down)
    Setting tau=0.7: V approximates E[Q | Q > median]
    Setting tau=0.9: V approximates E[Q | Q > 90th percentile] → near max

  - Three-network architecture (unique to IQL):
      1. V(s):      state value network (expectile-fitted)
      2. Q(s,a):    action-value network (Bellman backup using V, not max_a Q)
      3. pi(a|s):   actor (advantage-weighted behavior cloning)

  - Advantage-weighted regression (AWR) for actor:
        L_actor = -E[exp(beta * (Q(s,a) - V(s))) * log pi(a|s)]
    This is behavior cloning re-weighted by exp(Advantage).
    Intuition: actions that were BETTER than average in the dataset
    (Q > V → positive advantage) are upweighted in the BC objective.
    Actions that were worse than average are downweighted.

  - This completely avoids the OOD action problem:
    The actor only imitates dataset actions, just weighted by quality.
    The Q-function is only ever evaluated on *dataset* (s, a) pairs.

  - Advantage clipping: exp(beta * A) can explode when A is large.
    We clip the exponentiated advantage to a maximum value (exp_adv_max)
    to prevent a few high-advantage transitions from dominating training.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from copy import deepcopy

from src.utils.networks import TwinCritic, ValueNetwork, SquashedGaussianActor
from src.data.replay_buffer import Batch


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    """
    Asymmetric L2 (expectile) loss.

    Args:
        diff:      u = target - prediction  (positive = underpredicting)
        expectile: tau ∈ (0.5, 1.0), controls which quantile V approximates

    The standard MSE loss corresponds to tau=0.5 (symmetric).
    As tau→1, this loss makes V approximate the maximum of the target
    distribution, which is what we want for the Bellman backup.
    """
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


class IQL:
    """
    Implicit Q-Learning: fully in-support offline RL via expectile regression.

    Unlike CQL (which penalizes OOD Q-values) and TD3+BC (which regularizes
    toward dataset actions via MSE), IQL architecturally prevents any OOD
    action from being evaluated by decoupling policy extraction from Q-learning.

    Hyperparameters:
        expectile:   tau for V-network expectile regression (typically 0.7–0.9)
                     Higher = V approximates higher quantile of Q distribution
                     0.9 recommended for locomotion, 0.7 for antmaze
        temperature: beta for advantage-weighted actor (typically 0.1–10.0)
                     Higher = sharper weighting, closer to pure greedy policy
                     Lower  = smoother weighting, closer to pure BC
        exp_adv_max: clip exp(beta*A) to prevent single transitions dominating
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.7,
        temperature: float = 3.0,
        exp_adv_max: float = 100.0,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.discount = discount
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.exp_adv_max = exp_adv_max

        # ---- Networks ----
        self.actor  = SquashedGaussianActor(state_dim, action_dim, max_action=max_action).to(self.device)
        self.critic = TwinCritic(state_dim, action_dim, use_layer_norm=True).to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.value  = ValueNetwork(state_dim, use_layer_norm=True).to(self.device)

        # ---- Optimizers ----
        self.actor_optim  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.value_optim  = torch.optim.Adam(self.value.parameters(),  lr=value_lr)

        self._total_steps = 0

    # ------------------------------------------------------------------
    # Core update: three separate sub-updates
    # ------------------------------------------------------------------

    def update(self, batch: Batch) -> Dict[str, float]:
        """
        IQL update order:
          1. Update V(s) via expectile regression against Q_target(s,a_data)
          2. Update Q(s,a) via Bellman backup using V(s') (NOT max_a Q)
          3. Update actor via advantage-weighted behavior cloning
        """
        self._total_steps += 1
        metrics = {}

        metrics.update(self._update_value(batch))
        metrics.update(self._update_critic(batch))
        metrics.update(self._update_actor(batch))

        self._soft_update(self.critic, self.critic_target)

        return metrics

    # ------------------------------------------------------------------
    # Step 1: Value Network — Expectile Regression
    # ------------------------------------------------------------------

    def _update_value(self, batch: Batch) -> Dict[str, float]:
        """
        Fit V(s) to approximate a high quantile of Q(s, a_data).

        Target: V(s) ≈ E_{tau}[Q(s, a) | (s,a) ~ D]
        Loss:   L_V = expectile_loss(Q_target(s,a) - V(s), tau)

        Key: We use Q_TARGET (frozen) not Q (online) to compute the
        regression target. This prevents V and Q from chasing each other
        in a circular dependency.

        After convergence: V(s) ≈ quantile-tau of Q under behavior policy.
        For tau=0.7, V(s) is between the median and max of Q.
        For tau=0.9, V(s) is very close to max_a Q(s,a).
        """
        with torch.no_grad():
            q1_target, q2_target = self.critic_target(batch.observations, batch.actions)
            q_target = torch.min(q1_target, q2_target)

        v = self.value(batch.observations)
        value_loss = expectile_loss(q_target - v, self.expectile)

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        return {
            "value_loss": value_loss.item(),
            "v_mean":     v.mean().item(),
            "q_target_mean": q_target.mean().item(),
        }

    # ------------------------------------------------------------------
    # Step 2: Critic — Bellman Backup using V(s') instead of max_a Q(s',a')
    # ------------------------------------------------------------------

    def _update_critic(self, batch: Batch) -> Dict[str, float]:
        """
        Bellman backup with V-function bootstrap — the IQL innovation.

        Standard Q-learning bootstrap:
            y = r + gamma * max_{a'} Q(s', a')   ← requires querying OOD a'

        IQL bootstrap (OOD-free):
            y = r + gamma * V(s')                ← V is only fit on in-data (s,a)

        Since V(s') ≈ max_{a ~ D} Q(s', a), this is a *within-support* estimate
        of the next-state value. The policy improvement theorem still applies:
        any policy that achieves V(s) does at least as well as the behavior policy.
        """
        with torch.no_grad():
            v_next = self.value(batch.next_observations)
            td_target = batch.rewards + self.discount * (1.0 - batch.dones) * v_next

        q1, q2 = self.critic(batch.observations, batch.actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()

        return {
            "critic_loss": critic_loss.item(),
            "q1_mean":     q1.mean().item(),
            "td_target":   td_target.mean().item(),
        }

    # ------------------------------------------------------------------
    # Step 3: Actor — Advantage-Weighted Behavior Cloning (AWR)
    # ------------------------------------------------------------------

    def _update_actor(self, batch: Batch) -> Dict[str, float]:
        """
        AWR policy extraction: pi = argmax_pi E[exp(beta*A(s,a)) * log pi(a|s)]

        CRITICAL: log_prob must be evaluated at the DATASET action a_data,
        not at a freshly sampled action. The AWR objective is:
            L = -E_{(s,a)~D}[ w(s,a) * log pi(a | s) ]
        where w(s,a) = exp(beta * (Q(s,a) - V(s))).

        We use the SquashedGaussianActor's log_prob_of() to evaluate
        log pi(a_data | s) at the exact dataset action, with the tanh
        Jacobian correction applied. This keeps the actor entirely within
        the dataset support — no OOD actions are ever generated.
        """
        with torch.no_grad():
            q1, q2 = self.critic(batch.observations, batch.actions)
            q_min   = torch.min(q1, q2)
            v       = self.value(batch.observations)
            advantage = q_min - v                                      # (B, 1)
            exp_adv   = torch.exp(self.temperature * advantage).clamp(max=self.exp_adv_max)

        # Evaluate log pi(a_data | s) — must use dataset actions, not samples
        log_prob = self.actor.log_prob_of(batch.observations, batch.actions)  # (B, 1)

        # AWR loss: weighted negative log-likelihood on dataset actions
        actor_loss = -(exp_adv.detach() * log_prob).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optim.step()

        return {
            "actor_loss":     actor_loss.item(),
            "advantage_mean": advantage.mean().item(),
            "exp_adv_mean":   exp_adv.mean().item(),
            "log_prob_mean":  log_prob.mean().item(),
        }

    # ------------------------------------------------------------------
    # Target network
    # ------------------------------------------------------------------

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

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
            "value":         self.value.state_dict(),
            "total_steps":   self._total_steps,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.value.load_state_dict(ckpt["value"])
        self._total_steps = ckpt["total_steps"]