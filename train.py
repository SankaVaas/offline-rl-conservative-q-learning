"""
train.py — Main training script for offline-rl-conservative-q-learning.

Usage:
    python train.py --agent cql --dataset hopper-medium-v2 --steps 500000
    python train.py --agent iql --dataset halfcheetah-medium-v2 --steps 300000
    python train.py --agent td3bc --dataset walker2d-medium-v2 --steps 1000000

CPU Training note:
    All three agents are designed to run on CPU. Recommended batch size is
    256 (smaller than GPU runs of 1024) with proportionally fewer total steps.
    A 300k-step CQL run on hopper-medium takes ~45 minutes on modern CPU.
"""

import argparse
import os
import time
import numpy as np
import torch
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from src.agents.td3_bc import TD3BC
from src.agents.cql import CQL
from src.agents.iql import IQL
from src.data.replay_buffer import OfflineReplayBuffer
from src.utils.logger import Logger
from src.data.dataset import load_dataset

console = Console()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Offline RL Training")
    p.add_argument("--agent",    type=str, default="cql",
                   choices=["td3bc", "cql", "iql"], help="Agent algorithm")
    p.add_argument("--dataset",  type=str, default="hopper-medium-v2",
                   help="D4RL dataset name (or path to custom .npz)")
    p.add_argument("--steps",    type=int, default=300_000, help="Training steps")
    p.add_argument("--batch",    type=int, default=256,    help="Batch size")
    p.add_argument("--eval_freq",type=int, default=5_000,  help="Eval every N steps")
    p.add_argument("--eval_eps", type=int, default=5,      help="Episodes per eval (synthetic=5 is plenty)")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--tau",      type=float, default=0.005)
    p.add_argument("--save_dir", type=str, default="experiments/runs")
    # CQL-specific
    p.add_argument("--cql_alpha",    type=float, default=1.0)
    p.add_argument("--with_lagrange",action="store_true")
    # IQL-specific
    p.add_argument("--expectile",    type=float, default=0.7)
    p.add_argument("--temperature",  type=float, default=3.0)
    # TD3BC-specific
    p.add_argument("--bc_alpha",     type=float, default=2.5)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agent(args, state_dim: int, action_dim: int):
    common = dict(
        state_dim=state_dim,
        action_dim=action_dim,
        discount=args.discount,
        tau=args.tau,
        device="cpu",
    )
    if args.agent == "td3bc":
        return TD3BC(**common, alpha=args.bc_alpha)
    elif args.agent == "cql":
        return CQL(**common, cql_alpha=args.cql_alpha, with_lagrange=args.with_lagrange)
    elif args.agent == "iql":
        return IQL(**common, expectile=args.expectile, temperature=args.temperature)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # ---- Reproducibility ----
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Load dataset ----
    console.print(f"\n[bold]Loading dataset:[/bold] {args.dataset}")
    dataset, env_info = load_dataset(args.dataset)

    state_dim  = env_info["obs_dim"]
    action_dim = env_info["act_dim"]
    max_action = env_info["max_action"]

    # ---- Build offline buffer ----
    buffer = OfflineReplayBuffer(
        obs_dim=state_dim,
        action_dim=action_dim,
        normalize_obs=True,
        normalize_reward=True,
    )
    buffer.load_from_dict(dataset)

    # Print dataset statistics (demonstrates OOD problem scale)
    stats = buffer.dataset_statistics()
    table = Table(title="Dataset Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value",  style="magenta")
    for k, v in stats.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)

    # ---- Build agent ----
    agent = build_agent(args, state_dim, action_dim)
    console.print(f"\n[bold green]Agent:[/bold green] {agent}")

    # ---- Logger ----
    run_name = f"{args.agent}_{args.dataset}_{args.seed}_{int(time.time())}"
    log_dir  = os.path.join(args.save_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir=log_dir, agent_name=args.agent)

    # ---- Training loop ----
    console.print(f"\n[bold]Training for {args.steps:,} steps...[/bold]")
    console.print(f"Logs → {log_dir}\n")

    obs_normalizer = buffer.obs_normalizer()

    best_score = -np.inf
    pbar = tqdm(range(1, args.steps + 1), desc=f"{args.agent.upper()}", ncols=90)

    for step in pbar:
        # Sample minibatch from static offline buffer
        batch = buffer.sample(args.batch)

        # One gradient update
        metrics = agent.update(batch)
        logger.log_step(step, metrics)

        # ---- Periodic evaluation ----
        if step % args.eval_freq == 0:
            eval_score, eval_info = evaluate(
                agent=agent,
                dataset_name=args.dataset,
                n_episodes=args.eval_eps,
                obs_normalizer=obs_normalizer,
            )

            normalized = eval_info.get("normalized_score", eval_score)
            logger.log_eval(step, eval_score, normalized, eval_info)

            pbar.set_postfix({
                "score": f"{normalized:.1f}",
                "Q":     f"{metrics.get('q1_mean', metrics.get('q1_data_mean', 0)):.1f}",
            })

            # Save best checkpoint
            if normalized > best_score:
                best_score = normalized
                agent.save(os.path.join(log_dir, "best.pt"))

        # Save periodic checkpoint
        if step % (args.eval_freq * 10) == 0:
            agent.save(os.path.join(log_dir, f"step_{step}.pt"))

    console.print(f"\n[bold green]Training complete.[/bold green] Best score: {best_score:.1f}")
    console.print(f"Checkpoints saved → {log_dir}")

    # Final summary
    logger.summary()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(agent, dataset_name: str, n_episodes: int, obs_normalizer) -> tuple:
    """
    Evaluate the learned policy.

    For synthetic datasets: roll out in the LQR environment (instant, no gym needed).
    For real datasets: roll out in the gymnasium environment.

    Normalized score convention:
        0   = random policy performance
        100 = expert policy performance
    """
    if "synthetic" in dataset_name:
        return _evaluate_lqr(agent, dataset_name, n_episodes, obs_normalizer)
    return _evaluate_gym(agent, dataset_name, n_episodes, obs_normalizer)


def _evaluate_lqr(agent, dataset_name: str, n_episodes: int, obs_normalizer) -> tuple:
    """
    Evaluate on the synthetic LQR environment.

    LQR dynamics:
        s_{t+1} = 0.9*s + 0.1*a_padded + noise
        r_t     = -||s||^2 - 0.1*||a||^2

    Optimal LQR policy: a* = -0.5 * s[:act_dim]
    This gives us a reference score for normalization.

    We compute:
        random_score  : score of uniform random policy (Monte Carlo)
        optimal_score : score of closed-form LQR optimal policy
        agent_score   : score of learned policy
        normalized    : (agent - random) / (optimal - random) * 100
    """
    from src.data.dataset import ENV_SPECS
    key = next((k for k in ENV_SPECS if k in dataset_name), "hopper")
    obs_dim, act_dim, max_action = ENV_SPECS[key]

    gamma   = 0.99
    ep_len  = 1000
    rng     = np.random.default_rng(seed=999)

    def rollout(policy_fn):
        returns = []
        for _ in range(n_episodes):
            s = rng.standard_normal(obs_dim).astype(np.float32)
            ep_ret = 0.0
            discount = 1.0
            for _ in range(ep_len):
                a = policy_fn(s)
                a = np.clip(a, -max_action, max_action)
                noise = 0.01 * rng.standard_normal(obs_dim).astype(np.float32)
                a_pad = np.zeros(obs_dim, dtype=np.float32)
                a_pad[:act_dim] = a
                s_next = 0.9 * s + 0.1 * a_pad + noise
                r = -np.sum(s**2) - 0.1 * np.sum(a**2)
                ep_ret += discount * r
                discount *= gamma
                s = s_next
            returns.append(ep_ret)
        return float(np.mean(returns))

    # Agent policy (uses obs normalizer trained on dataset)
    def agent_policy(s):
        s_norm = obs_normalizer(s)
        return agent.select_action(s_norm)

    # Reference: optimal LQR policy
    def optimal_policy(s):
        return np.clip(-0.5 * s[:act_dim], -max_action, max_action)

    # Reference: random policy
    def random_policy(s):
        return rng.uniform(-max_action, max_action, act_dim).astype(np.float32)

    agent_score   = rollout(agent_policy)
    optimal_score = rollout(optimal_policy)
    random_score  = rollout(random_policy)

    # Normalized score: 0=random, 100=optimal
    denom = (optimal_score - random_score) + 1e-8
    normalized = (agent_score - random_score) / denom * 100.0

    return agent_score, {
        "raw_score":        agent_score,
        "normalized_score": normalized,
        "optimal_score":    optimal_score,
        "random_score":     random_score,
    }


def _evaluate_gym(agent, dataset_name: str, n_episodes: int, obs_normalizer) -> tuple:
    """Evaluate on real gymnasium environment (used with Minari datasets)."""
    try:
        import gymnasium as gym
    except ImportError:
        return 0.0, {"normalized_score": 0.0, "note": "gymnasium not installed"}

    mapping = {"hopper": "Hopper-v4", "walker2d": "Walker2d-v4",
               "halfcheetah": "HalfCheetah-v4", "ant": "Ant-v4"}
    env_name = next((v for k, v in mapping.items() if k in dataset_name), "Hopper-v4")

    try:
        env = gym.make(env_name)
    except Exception as e:
        return 0.0, {"normalized_score": 0.0, "note": str(e)}

    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, ep_ret = False, 0.0
        while not done:
            action = agent.select_action(obs_normalizer(obs))
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_ret += r
        returns.append(ep_ret)
    env.close()

    mean_ret = float(np.mean(returns))
    refs = {"hopper": (20.3, 3234.3), "walker2d": (1.6, 4592.3),
            "halfcheetah": (-280.2, 12135.0), "ant": (-325.6, 3879.7)}
    rand, expert = next((v for k, v in refs.items() if k in dataset_name), (0, 1))
    normalized = (mean_ret - rand) / (expert - rand + 1e-8) * 100.0

    return mean_ret, {"raw_score": mean_ret, "normalized_score": normalized}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)