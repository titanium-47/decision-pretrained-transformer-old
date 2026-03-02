"""
Evaluate a Decision Transformer on all darkroom-easy goals and plot a heatmap.
"""

import argparse
import os
import pathlib
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from collect_data import dagger_rollout
from envs.darkroom_env import DarkroomEnv, DarkroomEnvVec
from eval_policy import compute_episode_returns
from get_rollout_policy import get_rollout_policy
from models import DecisionTransformer


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_model_args_path(checkpoint_path):
    ckpt_path = pathlib.Path(checkpoint_path)
    search_dirs = []
    if ckpt_path.is_file():
        search_dirs.append(ckpt_path.parent)
    else:
        search_dirs.append(ckpt_path)
    search_dirs.append(search_dirs[0].parent)

    for directory in search_dirs:
        candidate = directory / "model_args.pkl"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find model_args.pkl near: {checkpoint_path}")


def _resolve_checkpoint_file(checkpoint_path):
    ckpt_path = pathlib.Path(checkpoint_path)
    if ckpt_path.is_file():
        return ckpt_path

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    epoch_checkpoints = sorted(
        ckpt_path.glob("model_epoch_*.pth"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if epoch_checkpoints:
        return epoch_checkpoints[-1]

    fallback_names = ["final_model.pth", "best_model.pth"]
    for name in fallback_names:
        candidate = ckpt_path / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No model checkpoint found in: {checkpoint_path}")


def load_model(checkpoint_path, device):
    model_args_path = _resolve_model_args_path(checkpoint_path)
    with open(model_args_path, "rb") as handle:
        model_args = pickle.load(handle)

    model = DecisionTransformer(model_args).to(device)
    checkpoint_file = _resolve_checkpoint_file(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    model.eval()
    return model, model_args, str(checkpoint_file)


def build_goal_envs(dim, horizon, goal_repeats, n_envs_per_batch):
    goals = [(x, y) for x in range(dim) for y in range(dim)]
    single_envs = []
    goal_indices = []
    for goal_idx, (x, y) in enumerate(goals):
        for _ in range(goal_repeats):
            single_envs.append(DarkroomEnv(dim, np.array([x, y]), horizon))
            goal_indices.append(goal_idx)

    if n_envs_per_batch <= 0:
        n_envs_per_batch = len(single_envs)

    vec_envs = []
    batch_goal_indices = []
    for i in range(0, len(single_envs), n_envs_per_batch):
        vec_envs.append(DarkroomEnvVec(single_envs[i : i + n_envs_per_batch]))
        batch_goal_indices.append(np.array(goal_indices[i : i + n_envs_per_batch], dtype=np.int32))
    return vec_envs, goals, batch_goal_indices


def aggregate_return(episode_returns, value_mode):
    if value_mode == "mean":
        return float(np.mean(episode_returns))
    if value_mode == "final_episode":
        return float(episode_returns[-1])
    raise ValueError(f"Unknown value_mode: {value_mode}")


def evaluate_goal_grid(
    model,
    vec_envs,
    goals,
    batch_goal_indices,
    env_horizon,
    eval_episodes,
    temp,
    context_horizon,
    sliding_window,
    value_mode,
):
    policy = get_rollout_policy(
        "decision_transformer",
        model=model,
        temp=temp,
        context_horizon=context_horizon,
        env_horizon=env_horizon,
        context_accumulation=False,
        sliding_window=sliding_window,
    )
    eval_horizon = eval_episodes * env_horizon
    dim = int(np.sqrt(len(goals)))
    heatmap = np.full((dim, dim), np.nan, dtype=np.float32)
    all_episode_returns = np.full((dim, dim, eval_episodes), 0.0, dtype=np.float32)

    rewards_batches = []
    for vec_env in tqdm.tqdm(vec_envs, desc="Collecting on-policy rollouts"):
        rollout_data = dagger_rollout(vec_env, policy, eval_horizon)
        rewards_batches.append(rollout_data["rewards"])
    rewards = np.concatenate(rewards_batches, axis=0)
    episode_returns, _, _ = compute_episode_returns(rewards, env_horizon)

    if value_mode == "mean":
        scalar_returns = np.mean(episode_returns, axis=1)
    elif value_mode == "final_episode":
        scalar_returns = episode_returns[:, -1]
    else:
        raise ValueError(f"Unknown value_mode: {value_mode}")

    goal_indices = np.concatenate(batch_goal_indices, axis=0)
    if len(goal_indices) != episode_returns.shape[0]:
        raise RuntimeError("Goal index mapping does not match collected trajectories.")

    goal_scalar_sum = np.zeros(len(goals), dtype=np.float64)
    goal_episode_sum = np.zeros((len(goals), eval_episodes), dtype=np.float64)
    goal_counts = np.zeros(len(goals), dtype=np.int32)

    for replica_idx, goal_idx in enumerate(tqdm.tqdm(goal_indices, desc="Aggregating goal repeats")):
        goal_scalar_sum[goal_idx] += scalar_returns[replica_idx]
        goal_episode_sum[goal_idx] += episode_returns[replica_idx]
        goal_counts[goal_idx] += 1

    if np.any(goal_counts == 0):
        raise RuntimeError("At least one goal has zero replicas after aggregation.")

    goal_scalar_mean = goal_scalar_sum / goal_counts
    goal_episode_mean = goal_episode_sum / goal_counts[:, None]
    for goal_idx, (x, y) in enumerate(goals):
        heatmap[y, x] = goal_scalar_mean[goal_idx]
        all_episode_returns[y, x, :] = goal_episode_mean[goal_idx]
    return heatmap, all_episode_returns


def plot_heatmap(heatmap, output_path, title, value_mode, eval_episodes, annotate_cells=False):
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(heatmap, cmap="viridis", origin="lower")
    colorbar = plt.colorbar(image, ax=ax)
    if value_mode == "mean":
        colorbar.set_label(f"Mean Return ({eval_episodes} episodes)")
    else:
        colorbar.set_label("Final Episode Return")

    ax.set_xticks(np.arange(heatmap.shape[1]))
    ax.set_yticks(np.arange(heatmap.shape[0]))
    ax.set_xlabel("Goal x")
    ax.set_ylabel("Goal y")
    ax.set_title(title)

    if annotate_cells:
        for y in range(heatmap.shape[0]):
            for x in range(heatmap.shape[1]):
                ax.text(x, y, f"{heatmap[y, x]:.1f}", ha="center", va="center", fontsize=6, color="white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot darkroom goal return heatmap")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint directory or .pth file")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--context_horizon",
        type=int,
        default=None,
        help="Transformer context horizon; default mimics train eval cap (min(model_horizon, 400)).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--annotate_cells", action="store_true")
    parser.add_argument("--value_mode", choices=["mean", "final_episode"], default="mean")
    parser.add_argument("--sliding_window", action="store_true")
    parser.add_argument("--goal_repeats", type=int, default=1, help="How many env replicas per goal")
    parser.add_argument(
        "--n_envs_per_batch",
        type=int,
        default=0,
        help="Vectorized env batch size. <=0 means run all goal replicas in one giant vec env.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, model_args, checkpoint_file = load_model(args.checkpoint_path, device)

    dim = 10
    env_horizon = 100
    vec_envs, goals, batch_goal_indices = build_goal_envs(
        dim=dim,
        horizon=env_horizon,
        goal_repeats=args.goal_repeats,
        n_envs_per_batch=args.n_envs_per_batch,
    )
    if args.context_horizon is not None:
        context_horizon = args.context_horizon
    else:
        # Match train_context_accumulator eval behavior where current_horizon is capped at 4 episodes.
        context_horizon = min(model_args["horizon"], 4 * env_horizon)

    heatmap, all_episode_returns = evaluate_goal_grid(
        model=model,
        vec_envs=vec_envs,
        goals=goals,
        batch_goal_indices=batch_goal_indices,
        env_horizon=env_horizon,
        eval_episodes=args.eval_episodes,
        temp=args.temp,
        context_horizon=context_horizon,
        sliding_window=args.sliding_window,
        value_mode=args.value_mode,
    )

    heatmap_path = os.path.join(args.output_dir, "darkroom_goal_returns_heatmap.png")
    heatmap_npy_path = os.path.join(args.output_dir, "darkroom_goal_returns.npy")
    heatmap_npz_path = os.path.join(args.output_dir, "darkroom_goal_returns.npz")

    title = (
        f"Darkroom 10x10 Goal Returns ({args.value_mode})\n"
        f"{os.path.basename(checkpoint_file)} | episodes={args.eval_episodes} | temp={args.temp}"
    )
    plot_heatmap(
        heatmap=heatmap,
        output_path=heatmap_path,
        title=title,
        value_mode=args.value_mode,
        eval_episodes=args.eval_episodes,
        annotate_cells=args.annotate_cells,
    )

    np.save(heatmap_npy_path, heatmap)
    np.savez(
        heatmap_npz_path,
        heatmap=heatmap,
        all_episode_returns=all_episode_returns,
        goals=np.array(goals, dtype=np.int32),
        checkpoint_file=checkpoint_file,
        eval_episodes=args.eval_episodes,
        env_horizon=env_horizon,
        context_horizon=context_horizon,
        temp=args.temp,
        value_mode=args.value_mode,
        sliding_window=args.sliding_window,
        seed=args.seed,
        goal_repeats=args.goal_repeats,
        n_envs_per_batch=args.n_envs_per_batch,
    )

    print(f"Saved heatmap image: {heatmap_path}")
    print(f"Saved heatmap array: {heatmap_npy_path}")
    print(f"Saved heatmap metrics: {heatmap_npz_path}")
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"All cells filled: {not np.isnan(heatmap).any()}")
    print(f"Goal repeats per cell: {args.goal_repeats}")
    print(f"Total env replicas evaluated: {len(goals) * args.goal_repeats}")


if __name__ == "__main__":
    main()
