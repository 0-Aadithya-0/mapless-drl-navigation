"""
Evaluate trained DQN checkpoints on the TurtleBot3 environment.

Exploration is disabled: the agent always takes its highest-Q action.

Evaluation runs in whichever world is currently launched. Pass --world
so the results record it. The Stage 3 test world has the training
arena's obstacles moving, so each episode meets a different layout:

    ros2 launch custom_world training.launch.py stage:=3 gui:=false

    python -m rl.evaluate --world stage3 \
        --checkpoint checkpoints/dqn_turtlebot_[0-9]*.pt

For each checkpoint the evaluation reports:

    - Success rate (episodes that reach max_steps without a collision
      and drive at least MIN_SUCCESS_DISTANCE)
    - Collision rate
    - Average reward, survival and distance driven
    - Step timing (sim seconds per step; 0.200 is healthy)

The summary table is ranked by average distance, then average reward;
the first row is the best checkpoint.

Results are also saved as JSON under runs/eval/.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from rl.agents.dqn import DQNAgent
from rl.utils.config import load_config
from turtlebot3_rl_nav.env import TurtleBotEnv

# Surviving is not enough: a policy that spins or stops in place never
# collides. A success must also drive this far forward (meters).
MIN_SUCCESS_DISTANCE = 1.0


def run_episode(
    env: TurtleBotEnv,
    agent: DQNAgent,
) -> dict:
    """Run one greedy episode and return its statistics."""

    state, _ = env.reset()

    done = False
    total_reward = 0.0
    distance = 0.0
    steps = 0
    step_durations = []
    info = {}

    while not done:
        action = agent.select_action(
            state,
            epsilon=0.0,
        )

        (
            state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        done = terminated or truncated

        total_reward += reward
        distance += info["progress"]
        steps += 1
        step_durations.append(info["step_duration"])

    collided = bool(info.get("collision", False))

    return {
        "reward": total_reward,
        "steps": steps,
        "collided": collided,
        "success": not collided and distance >= MIN_SUCCESS_DISTANCE,
        "distance": distance,
        "dt_sum": float(sum(step_durations)),
        "dt_max": float(max(step_durations)),
    }


def summarize(episodes: list[dict]) -> dict:
    """Aggregate per-episode statistics for one checkpoint."""

    n = len(episodes)
    total_steps = sum(e["steps"] for e in episodes)

    return {
        "episodes": n,
        "success_rate": sum(e["success"] for e in episodes) / n,
        "collision_rate": sum(e["collided"] for e in episodes) / n,
        "avg_reward": float(np.mean([e["reward"] for e in episodes])),
        "avg_steps": float(np.mean([e["steps"] for e in episodes])),
        "avg_distance": float(np.mean([e["distance"] for e in episodes])),
        "dt_mean": sum(e["dt_sum"] for e in episodes) / total_steps,
        "dt_max": max(e["dt_max"] for e in episodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained TurtleBot3 DQN checkpoints."
    )

    parser.add_argument(
        "--config",
        default="rl/configs/dqn.yaml",
        help="Path to the DQN configuration file.",
    )

    parser.add_argument(
        "--checkpoint",
        nargs="+",
        required=True,
        help="One or more checkpoints to evaluate, in order.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Evaluation episodes per checkpoint.",
    )

    parser.add_argument(
        "--world",
        required=True,
        help=(
            "Label of the launched world, e.g. stage3. Recorded in "
            "the results; it does not select the world."
        ),
    )

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1.")

    # Fail before Gazebo is touched, not halfway through a long run.
    missing = [p for p in args.checkpoint if not os.path.isfile(p)]

    if missing:
        parser.error(f"checkpoint not found: {', '.join(missing)}")

    # --------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------

    cfg = load_config(args.config)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # --------------------------------------------------------------
    # DQN agent
    # --------------------------------------------------------------

    # buffer_size matches training so full checkpoints, which carry
    # their replay buffer, always fit.
    agent = DQNAgent(
        state_dim=cfg["state_dim"],
        n_actions=cfg["n_actions"],
        buffer_size=cfg["buffer_size"],
        device=device,
    )

    # Load every GPU kernel now, before any step is being timed.
    agent.warmup()

    # --------------------------------------------------------------
    # Environment
    # --------------------------------------------------------------

    env = TurtleBotEnv(
        max_episode_steps=cfg["max_steps"],
    )

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if state_dim != cfg["state_dim"]:
        env.close()
        raise ValueError(
            f"State dimension mismatch: "
            f"environment={state_dim}, "
            f"config={cfg['state_dim']}"
        )

    if n_actions != cfg["n_actions"]:
        env.close()
        raise ValueError(
            f"Action count mismatch: "
            f"environment={n_actions}, "
            f"config={cfg['n_actions']}"
        )

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    results = {
        "world": args.world,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": args.config,
        "max_steps": cfg["max_steps"],
        "episodes_per_checkpoint": args.episodes,
        "checkpoints": {},
    }

    try:
        for checkpoint in args.checkpoint:
            agent.load(checkpoint)

            # Filled in place so an interrupt keeps finished episodes.
            episodes = []
            results["checkpoints"][checkpoint] = {
                "episodes": episodes,
            }

            print(f"\n==== {checkpoint} on {args.world} ====")

            for episode in range(1, args.episodes + 1):
                stats = run_episode(env, agent)
                episodes.append(stats)

                print(
                    f"ep {episode:3d} | "
                    f"reward {stats['reward']:8.2f} | "
                    f"survived {stats['steps']:4d} steps | "
                    f"collided {int(stats['collided'])} | "
                    f"distance {stats['distance']:5.2f} m | "
                    f"dt {stats['dt_sum'] / stats['steps']:.3f}/"
                    f"{stats['dt_max']:.3f}"
                )

    except KeyboardInterrupt:
        print("\nEvaluation interrupted; summarising finished episodes.")

    finally:
        env.close()

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    for checkpoint, entry in results["checkpoints"].items():
        if entry["episodes"]:
            entry["summary"] = summarize(entry["episodes"])

    # Best first: farthest average distance, ties broken by reward.
    summarized = sorted(
        (
            (checkpoint, entry["summary"])
            for checkpoint, entry in results["checkpoints"].items()
            if "summary" in entry
        ),
        key=lambda item: (
            item[1]["avg_distance"],
            item[1]["avg_reward"],
        ),
        reverse=True,
    )

    results["ranking"] = [checkpoint for checkpoint, _ in summarized]

    print(f"\n==== Evaluation on {args.world} (best first) ====")
    print(
        f"{'rank':>4s} {'checkpoint':34s} {'eps':>4s} {'success':>8s} "
        f"{'collide':>8s} {'reward':>7s} {'survival':>9s} "
        f"{'distance':>9s} {'dt':>11s}"
    )

    for rank, (checkpoint, s) in enumerate(summarized, start=1):
        print(
            f"{rank:4d} "
            f"{os.path.basename(checkpoint):34s} "
            f"{s['episodes']:4d} "
            f"{s['success_rate']:8.1%} "
            f"{s['collision_rate']:8.1%} "
            f"{s['avg_reward']:7.2f} "
            f"{s['avg_steps']:9.1f} "
            f"{s['avg_distance']:7.2f} m "
            f"{s['dt_mean']:.3f}/{s['dt_max']:.3f}"
        )

    if summarized:
        print(f"\nBest checkpoint: {summarized[0][0]}")

    os.makedirs("runs/eval", exist_ok=True)

    output_path = (
        f"runs/eval/eval_{args.world}_"
        f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
