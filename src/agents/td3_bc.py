"""
td3_bc.py — TD3+BC: Minimalist offline RL baseline.

Paper: "Minimalist Offline Reinforcement Learning" (Fujimoto & Gu, 2021)
       https://arxiv.org/abs/2106.06860

Theory exposed:
  - Behavior cloning (BC) as a policy constraint: instead of penalizing the
    Q-function for OOD actions, TD3+BC adds a BC term directly to the actor loss,
    pulling the policy towards the behavior policy mu(a|s) that generated the data.

  - The actor loss combines two objectives:
        L_actor = -lambda * Q(s, pi(s))  +  ||pi(s) - a_data||^2
      where lambda normalizes the Q-gradient scale so neither term dominates.

  - Lambda normalization (key insight): Raw Q-values can be very large (e.g., 200)
    while BC MSE loss is O(1). Without normalization, the Q-gradient would dominate
    and the BC term would have no effect. Lambda = alpha / (1/N * sum|Q(s,a)|)
    rescales Q-gradients to be O(1), balancing the two objectives.

  - Clipped Double Q (TD3): Two critics, Bellman target uses min(Q1,Q2) to
    reduce overestimation. Target networks (soft-updated) reduce bootstrapping
    instability — the 'deadly triad' of (function approximation, bootstrapping,
    off-policy data) that can diverge.

  - Target network soft update:
        theta_target <- tau * theta + (1 - tau) * theta_target
    Polyak averaging keeps target Q stable, preventing the 'moving target'
    problem in Bellman backup. Hard updates (tau=1) cause catastrophic
    oscillations in Q estimates.

  - Delayed policy update (TD3): Actor is updated every d critic steps.
    Motivation: if the critic is inaccurate, actor gradients are noisy.
    Letting the critic stabilize first reduces policy gradient variance.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from copy import deepcopy

from src.utils.networks import TwinCritic, DeterministicActor
from src.data.replay_buffer import Batch


class TD3BC:
    """
    TD3+BC: TD3 with Behavior Cloning regularization for offline RL.

    This is the *baseline* agent in this project. Despite its simplicity
    (just two extra lines vs. online TD3), it performs surprisingly well
    across many D4RL benchmarks — demonstrating that offline RL does not
    always need complex constraint mechanisms.

    Args:
        state_dim:         observation space dimension
        action_dim:        action space dimension
        max_action:        absolute maximum action magnitude
        discount:          reward discount factor gamma
        tau:               target network Polyak update rate
        policy_noise:      std of noise added to target policy actions
        noise_clip:        clipping range for target policy noise
        policy_freq:       actor update frequency (every N critic steps)
        alpha:             BC regularization weight (before lambda normalization)
        actor_lr:          actor learning rate
        critic_lr:         critic learning rate
        device:            cpu or cuda
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        alpha: float = 2.5,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        # ---- Networks ----
        self.actor = DeterministicActor(state_dim, action_dim, max_action=max_action).to(self.device)
        self.actor_target = deepcopy(self.actor)

        self.critic = TwinCritic(state_dim, action_dim, use_layer_norm=True).to(self.device)
        self.critic_target = deepcopy(self.critic)

        # ---- Optimizers ----
        self.actor_optim  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ---- Training state ----
        self._total_steps = 0
        self._train_metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, batch: Batch) -> Dict[str, float]:
        """
        One gradient update step on a sampled minibatch.

        Returns a dict of scalar metrics for logging.
        """
        self._total_steps += 1
        metrics = {}

        # ---- Step 1: Critic Update ----
        critic_metrics = self._update_critic(batch)
        metrics.update(critic_metrics)

        # ---- Step 2: Delayed Actor Update ----
        if self._total_steps % self.policy_freq == 0:
            actor_metrics = self._update_actor(batch)
            metrics.update(actor_metrics)
            self._soft_update(self.actor, self.actor_target)

        # ---- Step 3: Target Network Soft Update ----
        self._soft_update(self.critic, self.critic_target)

        self._train_metrics = metrics
        return metrics

    def _update_critic(self, batch: Batch) -> Dict[str, float]:
        """
        Bellman backup with clipped double Q and target policy smoothing.

        Target policy smoothing (TD3):
            a'_target = clip(pi_target(s') + eps, -max_a, max_a)
            eps ~ N(0, policy_noise), clipped to [-noise_clip, noise_clip]

        This smoothing regularizes the critic by preventing it from exploiting
        narrow peaks in Q(s', a') caused by the deterministic policy. Think of
        it as a form of data augmentation in action space.

        Bellman target:
            y = r + gamma * (1 - done) * min(Q1_target(s', a'), Q2_target(s', a'))

        Critic loss:
            L = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)
        """
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(batch.actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(batch.next_observations) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Bellman target with pessimistic Q estimate
            q1_target, q2_target = self.critic_target(batch.next_observations, next_action)
            q_target = torch.min(q1_target, q2_target)
            td_target = batch.rewards + self.discount * (1.0 - batch.dones) * q_target

        # Current Q estimates
        q1, q2 = self.critic(batch.observations, batch.actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        # Gradient clipping: prevents large Q-gradient steps that destabilize training
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()

        return {
            "critic_loss": critic_loss.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
            "td_target_mean": td_target.mean().item(),
        }

    def _update_actor(self, batch: Batch) -> Dict[str, float]:
        """
        TD3+BC actor loss = -lambda * Q(s, pi(s)) + ||pi(s) - a_data||^2

        Lambda normalization:
            lambda = alpha / mean(|Q(s, pi(s))|)
        This makes the Q-gradient component unit-scale, ensuring the BC
        regularization always has a meaningful effect.

        Without lambda:
          - If Q-values are large (e.g., 500), the gradient from -Q dominates.
          - BC term effectively vanishes, and the policy degenerates.

        With lambda:
          - The Q-gradient is rescaled to O(1).
          - BC term consistently regularizes toward the dataset actions.
          - alpha controls the tradeoff: higher alpha = more conservative.
        """
        pi = self.actor(batch.observations)
        q_pi = self.critic.q1(batch.observations, pi)  # use only Q1 for actor (TD3 convention)

        # Lambda normalization (detach to avoid gradient through normalization)
        lmbda = self.alpha / (q_pi.abs().mean().detach() + 1e-8)

        # BC regularization term: MSE between policy and dataset actions
        bc_loss = F.mse_loss(pi, batch.actions)

        # Combined loss: maximize Q (minimize -Q), minimize deviation from data
        actor_loss = -lmbda * q_pi.mean() + bc_loss

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optim.step()

        return {
            "actor_loss": actor_loss.item(),
            "bc_loss": bc_loss.item(),
            "q_pi_mean": q_pi.mean().item(),
            "lambda": lmbda.item(),
        }

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        """
        Polyak averaging: theta_target = tau*theta + (1-tau)*theta_target

        The intuition: a direct copy (tau=1) causes the Bellman target to
        jump discontinuously each time the critic updates, creating a
        'moving target' that the online network chases. Polyak averaging
        makes the target drift slowly and smoothly, providing a stable
        regression objective.

        Typical tau=0.005 means the target network has a half-life of
        ~140 updates (ln(0.5)/ln(0.995) ≈ 138).
        """
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Deterministic action selection for evaluation.
        Observation must be pre-normalized (use buffer.obs_normalizer()).
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor(obs_t)
        return action.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "actor":          self.actor.state_dict(),
            "actor_target":   self.actor_target.state_dict(),
            "critic":         self.critic.state_dict(),
            "critic_target":  self.critic_target.state_dict(),
            "total_steps":    self._total_steps,
        }, path)
        print(f"[TD3BC] Checkpoint saved → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self._total_steps = ckpt["total_steps"]
        print(f"[TD3BC] Checkpoint loaded ← {path} (step {self._total_steps:,})")

    def __repr__(self) -> str:
        return (
            f"TD3BC(alpha={self.alpha}, tau={self.tau}, "
            f"discount={self.discount}, policy_freq={self.policy_freq})"
        )