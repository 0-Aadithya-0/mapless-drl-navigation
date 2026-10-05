"""
Evaluate a trained DQN on the TurtleBot3 environment.

Exploration is disabled during evaluation.

The evaluation reports:

    - Average reward
    - Average survival time
    - Collision rate

Example:

    python -m rl.evaluate \
        --checkpoint checkpoints/dqn_turtlebot_final.pt

"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from rl.agents.dqn import DQNAgent
from rl.utils.config import load_config
from turtlebot3_rl_nav.env import TurtleBotEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained TurtleBot3 DQN."
    )

    parser.add_argument(
        "--config",
        default="rl/configs/dqn.yaml",
        help="Path to the DQN configuration file.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained DQN checkpoint.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes.",
    )

    args = parser.parse_args()

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
    # DQN agent
    # --------------------------------------------------------------

    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        device=device,
    )

    agent.load(args.checkpoint)

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    rewards = []
    episode_lengths = []
    collisions = 0

    try:
        for episode in range(
            1,
            args.episodes + 1,
        ):
            state, _ = env.reset()

            done = False
            episode_reward = 0.0
            episode_length = 0
            info = {}

            while not done:

                # epsilon=0 means:
                # always choose the highest-Q action.
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

                episode_reward += reward
                episode_length += 1

            rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if info.get("collision", False):
                collisions += 1

            print(
                f"ep {episode:3d} | "
                f"reward {episode_reward:8.2f} | "
                f"survived {episode_length:4d} steps | "
                f"collided "
                f"{int(info.get('collision', False))}"
            )

        # ----------------------------------------------------------
        # Summary
        # ----------------------------------------------------------

        n = args.episodes

        print("\n==== Evaluation ====")
        print(f"episodes:        {n}")
        print(
            f"avg reward:      "
            f"{np.mean(rewards):.2f}"
        )
        print(
            f"avg survival:    "
            f"{np.mean(episode_lengths):.2f} steps"
        )
        print(
            f"collision rate:  "
            f"{collisions / n:.1%}"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()

