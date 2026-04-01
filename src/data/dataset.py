"""
dataset.py — Offline dataset loading.

Supports:
  1. Minari datasets (D4RL-compatible, no MuJoCo license needed)
  2. Local .npz files (custom datasets)
  3. Synthetic dataset generator for CPU testing without any downloads

D4RL-style dataset dict format:
  {
    'observations':      np.ndarray (N, obs_dim)
    'actions':           np.ndarray (N, act_dim)
    'rewards':           np.ndarray (N,)
    'next_observations': np.ndarray (N, obs_dim)
    'terminals':         np.ndarray (N,) bool/int
  }
"""

import numpy as np
from typing import Dict, Tuple, Any


# Known environment specs (obs_dim, act_dim, max_action)
ENV_SPECS = {
    "hopper":      (11, 3, 1.0),
    "walker2d":    (17, 6, 1.0),
    "halfcheetah": (17, 6, 1.0),
    "ant":         (111, 8, 1.0),
    "pendulum":    (3,  1, 2.0),
}


def load_dataset(name: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load offline dataset by name or path.

    Priority:
      1. .npz file path (custom)
      2. Minari dataset
      3. Synthetic fallback (for testing without downloads)

    Returns:
        dataset:  D4RL-style dict
        env_info: {'obs_dim', 'act_dim', 'max_action'}
    """
    if name.endswith(".npz"):
        return _load_npz(name)

    # Try Minari
    try:
        return _load_minari(name)
    except Exception as e:
        print(f"[Dataset] Minari load failed ({e}). Using synthetic data.")
        return _synthetic_dataset(name)


def _load_minari(name: str) -> Tuple[Dict, Dict]:
    import minari
    dataset = minari.load_dataset(name)
    obs_dim = dataset.observation_space.shape[0]
    act_dim = dataset.action_space.shape[0]
    max_action = float(dataset.action_space.high.max())

    obs, acts, rews, next_obs, terms = [], [], [], [], []
    for ep in dataset.iterate_episodes():
        T = len(ep.rewards)
        obs.append(ep.observations[:-1])
        acts.append(ep.actions)
        rews.append(ep.rewards)
        next_obs.append(ep.observations[1:])
        terms.append(np.zeros(T, dtype=np.float32))
        terms[-1][-1] = 1.0  # mark episode end as terminal

    data = {
        "observations":      np.concatenate(obs),
        "actions":           np.concatenate(acts),
        "rewards":           np.concatenate(rews),
        "next_observations": np.concatenate(next_obs),
        "terminals":         np.concatenate(terms),
    }
    env_info = {"obs_dim": obs_dim, "act_dim": act_dim, "max_action": max_action}
    print(f"[Dataset] Loaded Minari '{name}': {len(data['rewards']):,} transitions.")
    return data, env_info


def _load_npz(path: str) -> Tuple[Dict, Dict]:
    d = np.load(path)
    data = {
        "observations":      d["observations"],
        "actions":           d["actions"],
        "rewards":           d["rewards"],
        "next_observations": d["next_observations"],
        "terminals":         d["terminals"],
    }
    obs_dim = data["observations"].shape[1]
    act_dim = data["actions"].shape[1]
    env_info = {"obs_dim": obs_dim, "act_dim": act_dim, "max_action": 1.0}
    return data, env_info


def _synthetic_dataset(name: str, n: int = 200_000) -> Tuple[Dict, Dict]:
    """
    Generate a synthetic offline dataset for CPU testing.

    The synthetic environment is a linear-quadratic regulator (LQR):
        s_{t+1} = 0.9 * s_t + 0.1 * a_t + noise
        r_t = -||s_t||^2 - 0.1 * ||a_t||^2

    This has a known optimal policy (LQR solution), making it useful for
    verifying that algorithms converge to something sensible without
    requiring MuJoCo or any gym environment.

    The 'medium' quality behavior policy mixes: 50% random, 50% noisy-optimal.
    """
    # Determine env spec from name
    key = next((k for k in ENV_SPECS if k in name), "hopper")
    obs_dim, act_dim, max_action = ENV_SPECS[key]

    print(f"[Dataset] Generating synthetic LQR dataset: n={n:,}, "
          f"obs={obs_dim}, act={act_dim}")

    rng = np.random.default_rng(seed=0)

    obs  = rng.standard_normal((n, obs_dim)).astype(np.float32)
    # Mixed behavior policy: partially random, partially optimal
    acts_random  = rng.uniform(-max_action, max_action, (n, act_dim)).astype(np.float32)
    acts_optimal = np.clip(-0.5 * obs[:, :act_dim], -max_action, max_action).astype(np.float32)
    mix = rng.binomial(1, 0.5, (n, 1)).astype(np.float32)
    acts = mix * acts_optimal + (1 - mix) * acts_random

    noise = 0.01 * rng.standard_normal((n, obs_dim)).astype(np.float32)
    next_obs = (0.9 * obs + 0.1 * np.pad(acts, ((0,0),(0, obs_dim - act_dim)))[:,:obs_dim] + noise).astype(np.float32)

    rews  = (-np.sum(obs**2, axis=1) - 0.1 * np.sum(acts**2, axis=1)).astype(np.float32)
    terms = np.zeros(n, dtype=np.float32)
    terms[np.arange(999, n, 1000)] = 1.0  # episode ends every 1000 steps

    data = {
        "observations":      obs,
        "actions":           acts,
        "rewards":           rews,
        "next_observations": next_obs,
        "terminals":         terms,
    }
    env_info = {"obs_dim": obs_dim, "act_dim": act_dim, "max_action": max_action}
    return data, env_info