
"""
Train a DQN on the TurtleBot3 obstacle-avoidance environment.

Run from the repository root:

    python -m rl.train

Training configuration is loaded from:

    rl/configs/dqn.yaml

The trained checkpoints are saved under:

    checkpoints/
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from rl.agents.dqn import DQNAgent
from rl.utils.config import load_config
from rl.utils.seeding import set_seed
from turtlebot3_rl_nav.env import TurtleBotEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DQN on TurtleBot3 obstacle avoidance."
    )

    parser.add_argument(
        "--config",
        default="rl/configs/dqn.yaml",
        help="Path to the DQN configuration file.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------

    cfg = load_config(args.config)

    set_seed(cfg["seed"])

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

    # Verify that configuration agrees with the environment contract.
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

    print(f"State dimension: {state_dim}")
    print(f"Number of actions: {n_actions}")

    # --------------------------------------------------------------
    # DQN agent
    # --------------------------------------------------------------

    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        gamma=cfg["gamma"],
        lr=cfg["lr"],
        buffer_size=cfg["buffer_size"],
        batch_size=cfg["batch_size"],
        target_update_every=cfg["target_update_every"],
        device=device,
    )

    # --------------------------------------------------------------
    # Logging / checkpoints
    # --------------------------------------------------------------

    writer = SummaryWriter(
        log_dir="runs/dqn_turtlebot"
    )

    os.makedirs("checkpoints", exist_ok=True)

    # --------------------------------------------------------------
    # Training state
    # --------------------------------------------------------------

    epsilon = cfg["eps_start"]
    total_steps = 0

    try:
        # ----------------------------------------------------------
        # Episodes
        # ----------------------------------------------------------

        for episode in range(
            1,
            cfg["max_episodes"] + 1,
        ):
            state, _ = env.reset()

            done = False
            episode_reward = 0.0
            episode_length = 0
            collided = False
            episode_losses = []

            # ------------------------------------------------------
            # Episode loop
            # ------------------------------------------------------

            while not done:

                # Initial experience collection.
                if total_steps < cfg["learning_starts"]:
                    action = env.action_space.sample()

                else:
                    action = agent.select_action(
                        state,
                        epsilon,
                    )

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                done = terminated or truncated

                # Store the REAL terminal condition.
                #
                # Collision = terminated
                # Timeout   = truncated
                #
                # We intentionally store only terminated here.
                agent.buffer.add(
                    state,
                    action,
                    reward,
                    next_state,
                    float(terminated),
                )

                # Begin learning after the warm-up period.
                if total_steps >= cfg["learning_starts"]:
                    loss = agent.learn()

                    if loss is not None:
                        episode_losses.append(loss)

                state = next_state

                episode_reward += reward
                episode_length += 1
                total_steps += 1

                if info.get("collision", False):
                    collided = True

            # ------------------------------------------------------
            # Exploration decay
            # ------------------------------------------------------

            epsilon = max(
                cfg["eps_end"],
                epsilon * cfg["eps_decay"],
            )

            # ------------------------------------------------------
            # Logging
            # ------------------------------------------------------

            writer.add_scalar(
                "reward/episode",
                episode_reward,
                episode,
            )

            writer.add_scalar(
                "survival_steps",
                episode_length,
                episode,
            )

            writer.add_scalar(
                "collision",
                int(collided),
                episode,
            )

            writer.add_scalar(
                "epsilon",
                epsilon,
                episode,
            )

            if episode_losses:
                writer.add_scalar(
                    "loss/episode",
                    sum(episode_losses)
                    / len(episode_losses),
                    episode,
                )

            print(
                f"ep {episode:4d} | "
                f"reward {episode_reward:8.2f} | "
                f"survived {episode_length:4d} steps | "
                f"collided {int(collided)} | "
                f"eps {epsilon:.3f}"
            )

            # ------------------------------------------------------
            # Periodic checkpoint
            # ------------------------------------------------------

            if episode % cfg["save_every"] == 0:
                checkpoint_path = (
                    f"checkpoints/"
                    f"dqn_turtlebot_{episode}.pt"
                )

                agent.save(checkpoint_path)

                print(
                    f"Saved checkpoint: "
                    f"{checkpoint_path}"
                )

        # ----------------------------------------------------------
        # Final checkpoint
        # ----------------------------------------------------------

        final_checkpoint = (
            "checkpoints/dqn_turtlebot_final.pt"
        )

        agent.save(final_checkpoint)

        print(
            f"\nTraining complete.\n"
            f"Final checkpoint: {final_checkpoint}"
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

        interrupted_checkpoint = (
            "checkpoints/dqn_turtlebot_interrupted.pt"
        )

        agent.save(interrupted_checkpoint)

        print(
            f"Saved interrupted checkpoint: "
            f"{interrupted_checkpoint}"
        )

    finally:
        env.close()
        writer.close()


if __name__ == "__main__":
    main()

