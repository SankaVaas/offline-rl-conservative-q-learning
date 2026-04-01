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
    p.add_argument("--eval_eps", type=int, default=10,     help="Episodes per eval")
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
    Roll out the learned policy in the environment for n_episodes.

    The normalized score follows the D4RL convention:
        normalized = (score - random_score) / (expert_score - random_score) * 100
    This puts all tasks on a common scale where 0=random, 100=expert.
    """
    try:
        import gymnasium as gym
    except ImportError:
        return 0.0, {"normalized_score": 0.0, "raw_score": 0.0, "note": "gymnasium not found"}

    env_name = _dataset_to_env(dataset_name)

    try:
        env = gym.make(env_name)
    except Exception as e:
        return 0.0, {"normalized_score": 0.0, "note": str(e)}

    episode_returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            obs_norm = obs_normalizer(obs)
            action = agent.select_action(obs_norm)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
        episode_returns.append(ep_return)

    env.close()

    mean_return = np.mean(episode_returns)
    std_return  = np.std(episode_returns)

    # D4RL normalized score reference values (approximate)
    normalized = _normalize_score(dataset_name, mean_return)

    return mean_return, {
        "raw_score":        mean_return,
        "score_std":        std_return,
        "normalized_score": normalized,
    }


def _dataset_to_env(dataset_name: str) -> str:
    """Map D4RL dataset name to gymnasium env name."""
    mapping = {
        "hopper":      "Hopper-v4",
        "walker2d":    "Walker2d-v4",
        "halfcheetah": "HalfCheetah-v4",
        "ant":         "Ant-v4",
    }
    for key, env in mapping.items():
        if key in dataset_name:
            return env
    return "Hopper-v4"


def _normalize_score(dataset_name: str, raw_score: float) -> float:
    """
    D4RL normalized score: (score - random) / (expert - random) * 100
    Reference scores from Fu et al. (2020).
    """
    # (random_score, expert_score) pairs
    refs = {
        "hopper":             (20.3,   3234.3),
        "walker2d":           (1.6,    4592.3),
        "halfcheetah":        (-280.2, 12135.0),
        "ant":                (-325.6, 3879.7),
    }
    for key, (rand, expert) in refs.items():
        if key in dataset_name:
            return (raw_score - rand) / (expert - rand) * 100.0
    return raw_score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)