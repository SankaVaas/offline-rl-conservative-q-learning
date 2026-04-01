"""
replay_buffer.py — Static offline replay buffer.

Theory exposed:
  - The fundamental shift from online to offline RL: the buffer is FIXED.
    No new (s, a, r, s') tuples are ever added. The agent must extract a
    good policy purely from historical data — analogous to supervised learning
    from a logged dataset.

  - Distribution shift: The data was collected by some *behavior policy* mu(a|s).
    Any policy pi(a|s) trained from this data may query state-action pairs
    (s, a') where a' was NEVER taken by mu. Q-values for these OOD (out-of-
    distribution) actions are extrapolated by the critic, often wildly
    overestimated. This is the core challenge offline RL must solve.

  - Dataset coverage: Denoted C(s,a) = 1 if (s,a) appears in the dataset.
    Offline RL algorithms differ in how they handle low-coverage regions:
      * TD3+BC:  penalizes actions far from dataset via BC regularization
      * CQL:     adds a penalty that *lower-bounds* Q on OOD actions
      * IQL:     avoids OOD actions entirely — never queries Q(s,a') for a' ~ pi

  - Normalization: Reward and observation normalization are standard practice.
    Un-normalized rewards cause Q-value scales to vary wildly across tasks,
    making hyperparameters non-transferable. Obs normalization improves
    gradient conditioning for the critic's input layer.
"""

import numpy as np
import torch
from typing import Dict, NamedTuple, Optional
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Batch type — typed container for a sampled minibatch
# ---------------------------------------------------------------------------

@dataclass
class Batch:
    """
    A single sampled minibatch from the offline buffer.

    All tensors are float32, shape: (batch_size, dim).
    'done' is float in {0.0, 1.0} rather than bool to allow direct use in
    the Bellman target:
        y = r + gamma * (1 - done) * V(s')
    """
    observations:      torch.Tensor   # (B, obs_dim)
    actions:           torch.Tensor   # (B, act_dim)
    rewards:           torch.Tensor   # (B, 1)
    next_observations: torch.Tensor   # (B, obs_dim)
    dones:             torch.Tensor   # (B, 1)  float {0, 1}


# ---------------------------------------------------------------------------
# Offline Replay Buffer
# ---------------------------------------------------------------------------

class OfflineReplayBuffer:
    """
    Immutable replay buffer for offline reinforcement learning.

    Unlike online RL buffers (e.g. DQN's experience replay), this buffer
    is loaded ONCE and never written to again. All 'experience' comes
    from a pre-collected dataset (e.g., D4RL, Minari, or custom logs).

    Key design choices:
      1. Store everything as float32 numpy arrays — CPU-friendly, avoids
         repeated dtype casting during sampling.
      2. Pre-normalize observations and rewards at load time — constants
         are stored so evaluation rollouts can use the same normalization.
      3. Uniform random sampling — no prioritized experience replay (PER)
         since offline RL already has a fixed data distribution; PER would
         further bias away from the behavior policy distribution.

    Args:
        obs_dim:         dimensionality of observation space
        action_dim:      dimensionality of action space
        max_size:        maximum number of transitions to store
        device:          torch device for sampled batches
        normalize_obs:   whether to z-score observations (mean=0, std=1)
        normalize_reward: whether to normalize rewards to [-1, 1] range
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_size: int = 1_000_000,
        device: str = "cpu",
        normalize_obs: bool = True,
        normalize_reward: bool = True,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.device = torch.device(device)
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward

        # Pre-allocate storage
        self._obs   = np.zeros((max_size, obs_dim),    dtype=np.float32)
        self._acts  = np.zeros((max_size, action_dim), dtype=np.float32)
        self._rews  = np.zeros((max_size, 1),          dtype=np.float32)
        self._next  = np.zeros((max_size, obs_dim),    dtype=np.float32)
        self._dones = np.zeros((max_size, 1),          dtype=np.float32)

        self._size = 0

        # Normalization statistics (set after load)
        self.obs_mean:  Optional[np.ndarray] = None
        self.obs_std:   Optional[np.ndarray] = None
        self.rew_min:   Optional[float] = None
        self.rew_max:   Optional[float] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_dict(self, dataset: Dict[str, np.ndarray]) -> None:
        """
        Load a D4RL-style dataset dict with keys:
            'observations', 'actions', 'rewards',
            'next_observations', 'terminals'

        This is the standard format used by D4RL (Fu et al., 2020) and
        Minari. The buffer stores up to max_size transitions; if the
        dataset is larger, it is truncated (rare in practice).
        """
        n = len(dataset["observations"])
        n = min(n, self.max_size)

        self._obs[:n]   = dataset["observations"][:n].astype(np.float32)
        self._acts[:n]  = dataset["actions"][:n].astype(np.float32)
        self._rews[:n]  = dataset["rewards"][:n].reshape(-1, 1).astype(np.float32)
        self._next[:n]  = dataset["next_observations"][:n].astype(np.float32)
        self._dones[:n] = dataset["terminals"][:n].reshape(-1, 1).astype(np.float32)
        self._size = n

        print(f"[Buffer] Loaded {n:,} transitions. "
              f"obs_dim={self.obs_dim}, act_dim={self.action_dim}")

        self._compute_normalization_stats()
        if self.normalize_obs:
            self._apply_obs_normalization()
        if self.normalize_reward:
            self._apply_reward_normalization()

    def load_from_arrays(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Convenience loader for custom array format."""
        dataset = {
            "observations": obs,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_obs,
            "terminals": dones,
        }
        self.load_from_dict(dataset)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _compute_normalization_stats(self) -> None:
        """
        Compute statistics over the full offline dataset.

        Note: we compute stats *before* normalization so the stored
        constants can be used to un-normalize during evaluation.
        """
        obs = self._obs[:self._size]
        rew = self._rews[:self._size]

        self.obs_mean = obs.mean(axis=0)
        self.obs_std  = obs.std(axis=0) + 1e-8  # avoid division by zero

        self.rew_min  = float(rew.min())
        self.rew_max  = float(rew.max())

        print(f"[Buffer] Reward range: [{self.rew_min:.3f}, {self.rew_max:.3f}]")
        print(f"[Buffer] Obs mean norm: {np.linalg.norm(self.obs_mean):.3f}")

    def _apply_obs_normalization(self) -> None:
        """
        Z-score normalize observations in-place.

        Motivation: Q-networks receive raw observations as input. If features
        have vastly different scales (e.g., position in meters vs. velocity in
        m/s), the first linear layer effectively learns a poorly conditioned
        weight matrix. Normalization makes all input features unit-variance,
        which improves gradient flow and reduces sensitivity to learning rate.
        """
        self._obs[:self._size]  = (self._obs[:self._size] - self.obs_mean) / self.obs_std
        self._next[:self._size] = (self._next[:self._size] - self.obs_mean) / self.obs_std
        print("[Buffer] Observations z-score normalized.")

    def _apply_reward_normalization(self) -> None:
        """
        Normalize rewards to [-1, 1] using min-max scaling.

        Motivation: The scale of rewards directly determines the scale of
        Q-values (since Q = sum of discounted rewards). Large reward scales
        cause large Q-values which destabilize training. Normalizing to [-1,1]
        makes the Q-value magnitude predictable regardless of the environment.

        Alternative: divide by (rew_max - rew_min) to get [0, 1] range —
        used in some implementations. We use [-1, 1] to handle negative rewards.
        """
        r_range = (self.rew_max - self.rew_min) + 1e-8
        self._rews[:self._size] = (
            2.0 * (self._rews[:self._size] - self.rew_min) / r_range - 1.0
        )
        print("[Buffer] Rewards normalized to [-1, 1].")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> Batch:
        """
        Uniformly sample a minibatch of transitions.

        Uniform sampling means the empirical batch distribution matches the
        behavior policy's state-action visitation frequency d^mu(s,a).
        This is intentional: all offline RL algorithms implicitly or explicitly
        constrain the learned policy to stay close to d^mu.

        Returns a Batch with all tensors on self.device.
        """
        idx = np.random.randint(0, self._size, size=batch_size)

        return Batch(
            observations=self._to_tensor(self._obs[idx]),
            actions=self._to_tensor(self._acts[idx]),
            rewards=self._to_tensor(self._rews[idx]),
            next_observations=self._to_tensor(self._next[idx]),
            dones=self._to_tensor(self._dones[idx]),
        )

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(arr).to(self.device)

    # ------------------------------------------------------------------
    # Properties & utilities
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    def dataset_statistics(self) -> Dict[str, float]:
        """
        Compute dataset-level statistics useful for diagnosing data quality.

        Returns metrics including:
          - action_coverage: std of actions (higher = more diverse behavior)
          - terminal_rate:   fraction of done=1 transitions
          - mean_reward:     average reward in dataset
        """
        acts  = self._acts[:self._size]
        rews  = self._rews[:self._size]
        dones = self._dones[:self._size]

        return {
            "n_transitions":   self._size,
            "mean_reward":     float(rews.mean()),
            "std_reward":      float(rews.std()),
            "terminal_rate":   float(dones.mean()),
            "action_mean_norm": float(np.linalg.norm(acts.mean(axis=0))),
            "action_std_mean":  float(acts.std(axis=0).mean()),
        }

    def obs_normalizer(self):
        """
        Return a callable that normalizes raw observations at eval time,
        matching the normalization applied during training.
        """
        mean = self.obs_mean
        std  = self.obs_std

        def normalize(obs: np.ndarray) -> np.ndarray:
            return (obs - mean) / std

        return normalize if self.normalize_obs else lambda x: x

    def __repr__(self) -> str:
        return (
            f"OfflineReplayBuffer("
            f"size={self._size:,}, "
            f"obs_dim={self.obs_dim}, "
            f"act_dim={self.action_dim}, "
            f"normalize_obs={self.normalize_obs}, "
            f"normalize_reward={self.normalize_reward})"
        )