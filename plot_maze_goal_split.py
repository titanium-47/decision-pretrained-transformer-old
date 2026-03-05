"""
Plot maze goal splits and per-goal success/failure maps.

Uses the same maze loading + goal split path as train_context_accumulator.py:
maze_env.load_maze_and_goals(maze_path, train_ratio, seed).
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from get_rollout_policy import TransformerCNNPolicy
from maze_env import VecMazeEnv, load_maze_and_goals
from models import DecisionTransformerCnn


def load_nav_and_goals(maze_path: str, train_goal_ratio: float, seed: int):
    """Load maze + train/eval goals exactly like training."""
    nav, train_goals, eval_goals = load_maze_and_goals(
        maze_path=maze_path,
        train_ratio=train_goal_ratio,
        seed=seed,
    )
    return nav, train_goals, eval_goals


def plot_goal_split(nav: np.ndarray, train_goals, eval_goals, output_path: Path):
    """Plot maze walls/free cells with train/eval goal overlays."""
    fig, ax = plt.subplots(figsize=(7, 7))

    # Walls are dark, free cells light.
    ax.imshow(~nav, cmap="gray_r", interpolation="nearest", origin="upper")

    train_arr = np.array(train_goals)
    eval_arr = np.array(eval_goals)

    ax.scatter(
        train_arr[:, 0],
        train_arr[:, 1],
        s=18,
        c="#1f77b4",
        marker="o",
        alpha=0.85,
        label=f"Train goals ({len(train_goals)})",
    )
    ax.scatter(
        eval_arr[:, 0],
        eval_arr[:, 1],
        s=26,
        c="#d62728",
        marker="x",
        alpha=0.95,
        label=f"Eval goals ({len(eval_goals)})",
    )

    ax.set_title("Maze Goal Split: Train vs Eval")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, nav.shape[1] - 0.5)
    ax.set_ylim(nav.shape[0] - 0.5, -0.5)
    ax.grid(False)
    ax.legend(loc="upper right", frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _load_cnn_policy(checkpoint_path, model_args_path, device, temp):
    """Load DecisionTransformerCnn + TransformerCNNPolicy for rollout."""
    with open(model_args_path, "rb") as f:
        model_args = pickle.load(f)
    model = DecisionTransformerCnn(model_args).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    context_horizon = int(model_args.get("horizon", 1000))
    policy = TransformerCNNPolicy(model=model, temp=temp, context_horizon=context_horizon)
    obs_shape = tuple(model_args.get("obs", ()))
    if len(obs_shape) != 3:
        raise ValueError(f"Unexpected model obs shape in model_args: {obs_shape}")
    return policy, obs_shape


def _run_goal_trials(
    nav,
    goals,
    policy_type,
    trials_per_goal,
    max_steps,
    rng,
    visibility,
    cnn_policy=None,
):
    """Evaluate a policy on each goal using VecMazeEnv; returns per-goal success rates."""
    results = {}
    for goal in goals:
        # Each goal is fixed; trials run in parallel over vectorized envs.
        env = VecMazeEnv(
            nav=nav,
            goal_cells=[goal],
            n_envs=trials_per_goal,
            visibility=visibility,
            max_steps=max_steps,
        )
        obs, infos = env.reset()
        n = env.n
        pending = np.ones(n, dtype=bool)
        successes = np.zeros(n, dtype=bool)
        if policy_type == "cnn":
            cnn_policy.reset(np.ones(n, dtype=bool))

        for _ in range(max_steps):
            if not np.any(pending):
                break

            if policy_type == "expert":
                actions = env.expert_actions()
            elif policy_type == "random":
                actions = rng.randint(0, 4, size=n)
            elif policy_type == "cnn":
                actions = cnn_policy.get_action(obs, infos)
            else:
                raise ValueError(f"Unknown policy_type: {policy_type}")

            next_obs, rewards, dones, next_infos = env.step(actions)

            if policy_type == "cnn":
                cnn_policy.update_context(obs, infos, actions, rewards, dones)
                cnn_policy.reset(dones)

            newly_done = pending & dones
            successes[newly_done] = rewards[newly_done] > 0
            pending[newly_done] = False

            obs, infos = next_obs, next_infos

        results[goal] = float(np.mean(successes.astype(np.float32)))
    return results


def _split_by_success(goals, success_rates, threshold):
    success_goals, fail_goals = [], []
    for g in goals:
        if success_rates[g] >= threshold:
            success_goals.append(g)
        else:
            fail_goals.append(g)
    return success_goals, fail_goals


def _scatter_if_nonempty(ax, pts, **kwargs):
    if len(pts) == 0:
        return
    arr = np.array(pts)
    ax.scatter(arr[:, 0], arr[:, 1], **kwargs)


def plot_success_fail_map(
    nav,
    train_success,
    train_fail,
    eval_success,
    eval_fail,
    output_path,
    title,
):
    """Plot maze with train/eval success/failure goal categories."""
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(~nav, cmap="gray_r", interpolation="nearest", origin="upper")

    _scatter_if_nonempty(
        ax,
        train_success,
        s=26,
        c="#2ca02c",
        marker="o",
        alpha=0.85,
        label=f"Train success ({len(train_success)})",
    )
    _scatter_if_nonempty(
        ax,
        train_fail,
        s=30,
        c="#d62728",
        marker="o",
        facecolors="none",
        linewidths=1.5,
        alpha=0.95,
        label=f"Train fail ({len(train_fail)})",
    )
    _scatter_if_nonempty(
        ax,
        eval_success,
        s=40,
        c="#17becf",
        marker="x",
        alpha=0.95,
        label=f"Eval success ({len(eval_success)})",
    )
    _scatter_if_nonempty(
        ax,
        eval_fail,
        s=45,
        c="#ff7f0e",
        marker="x",
        alpha=0.95,
        label=f"Eval fail ({len(eval_fail)})",
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, nav.shape[1] - 0.5)
    ax.set_ylim(nav.shape[0] - 0.5, -0.5)
    ax.grid(False)
    ax.legend(loc="upper right", frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maze_path",
        type=str,
        default="maze.npy",
        help="Path to maze .npy file (same input used by maze_env.load_maze_and_goals).",
    )
    parser.add_argument("--train_goal_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="plots/maze_goal_split.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="split",
        choices=["split", "success_fail"],
        help="split: train/eval goals only. success_fail: rollout each goal and plot outcomes.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="expert",
        choices=["expert", "random", "cnn"],
        help="Policy used in success_fail mode.",
    )
    parser.add_argument("--trials_per_goal", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--success_threshold", type=float, default=1.0)
    parser.add_argument("--rollout_seed", type=int, default=42)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--model_args_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument(
        "--visibility",
        type=int,
        default=7,
        help="Observation window size for env rollouts (matches training default).",
    )
    args = parser.parse_args()

    nav, train_goals, eval_goals = load_nav_and_goals(
        maze_path=args.maze_path,
        train_goal_ratio=args.train_goal_ratio,
        seed=args.seed,
    )

    output_path = Path(args.output)

    if args.mode == "split":
        plot_goal_split(nav, train_goals, eval_goals, output_path)
        print(
            f"Saved plot to {output_path} "
            f"(train={len(train_goals)}, eval={len(eval_goals)}, free={len(train_goals)+len(eval_goals)})"
        )
        return

    rng = np.random.RandomState(args.rollout_seed)
    cnn_policy = None
    rollout_visibility = args.visibility
    if args.policy == "cnn":
        if args.checkpoint_path is None or args.model_args_path is None:
            raise ValueError(
                "--checkpoint_path and --model_args_path are required when --policy cnn"
            )
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        cnn_policy, obs_shape = _load_cnn_policy(
            checkpoint_path=args.checkpoint_path,
            model_args_path=args.model_args_path,
            device=device,
            temp=args.temp,
        )
        # Ensure env observation shape matches the model's expected spatial size.
        model_visibility = int(obs_shape[0])
        if rollout_visibility != model_visibility:
            print(
                f"[info] Overriding visibility {rollout_visibility} -> "
                f"{model_visibility} to match model obs shape {obs_shape}."
            )
            rollout_visibility = model_visibility

    train_rates = _run_goal_trials(
        nav=nav,
        goals=train_goals,
        policy_type=args.policy,
        trials_per_goal=args.trials_per_goal,
        max_steps=args.max_steps,
        rng=rng,
        visibility=rollout_visibility,
        cnn_policy=cnn_policy,
    )
    eval_rates = _run_goal_trials(
        nav=nav,
        goals=eval_goals,
        policy_type=args.policy,
        trials_per_goal=args.trials_per_goal,
        max_steps=args.max_steps,
        rng=rng,
        visibility=rollout_visibility,
        cnn_policy=cnn_policy,
    )

    train_success, train_fail = _split_by_success(
        train_goals, train_rates, args.success_threshold
    )
    eval_success, eval_fail = _split_by_success(
        eval_goals, eval_rates, args.success_threshold
    )

    title = (
        f"Maze Goal Outcomes ({args.policy}, trials={args.trials_per_goal}, "
        f"threshold={args.success_threshold:.2f})"
    )
    plot_success_fail_map(
        nav=nav,
        train_success=train_success,
        train_fail=train_fail,
        eval_success=eval_success,
        eval_fail=eval_fail,
        output_path=output_path,
        title=title,
    )

    print(f"Saved success/fail plot to {output_path}")
    print(
        f"Train: {len(train_success)} success, {len(train_fail)} fail | "
        f"Eval: {len(eval_success)} success, {len(eval_fail)} fail"
    )


if __name__ == "__main__":
    main()
