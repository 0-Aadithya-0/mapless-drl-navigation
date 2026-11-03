
"""
Train a DQN on the TurtleBot3 obstacle-avoidance environment.

Run from the repository root:

    python -m rl.train

Training configuration is loaded from:

    rl/configs/dqn.yaml

The trained checkpoints are saved under:

    checkpoints/

Resume from a saved checkpoint:

    python -m rl.train --resume checkpoints/dqn_turtlebot_100.pt

Checkpoints hold the full training state (networks, optimizer, replay
buffer, episode, epsilon), so a resumed run continues where it stopped.
Older checkpoints hold only Q-network weights and resume as a warm
start; their episode number is read from a dqn_turtlebot_<episode>.pt
filename, or given with --start-episode.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import time

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

    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint to continue training from.",
    )

    parser.add_argument(
        "--start-episode",
        type=int,
        default=None,
        help=(
            "Episodes already completed by a weight-only --resume "
            "checkpoint. Defaults to the number in its filename."
        ),
    )

    args = parser.parse_args()

    if args.start_episode is not None and args.resume is None:
        parser.error("--start-episode requires --resume.")

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
    # DQN agent
    # --------------------------------------------------------------

    # Built from the config so a checkpoint is checked before the
    # environment starts; the environment contract check below
    # confirms these shapes.
    agent = DQNAgent(
        state_dim=cfg["state_dim"],
        n_actions=cfg["n_actions"],
        gamma=cfg["gamma"],
        lr=cfg["lr"],
        buffer_size=cfg["buffer_size"],
        batch_size=cfg["batch_size"],
        target_update_every=cfg["target_update_every"],
        device=device,
    )

    # Load every GPU kernel now, before any step is being timed.
    agent.warmup()

    # --------------------------------------------------------------
    # Resume
    # --------------------------------------------------------------

    start_episode = 0
    epsilon = cfg["eps_start"]
    total_steps = 0
    random_warmup = True

    if args.resume is not None:
        saved = agent.load(args.resume)

        if saved is not None:
            # Full checkpoint: continue exactly where it stopped.
            if args.start_episode is not None:
                parser.error(
                    "--start-episode only applies to weight-only "
                    "checkpoints; this one records its episode."
                )

            start_episode = saved["episode"]
            epsilon = saved["epsilon"]
            total_steps = saved["total_steps"]

        else:
            # Weight-only checkpoint: warm start.
            if args.start_episode is not None:
                start_episode = args.start_episode
            else:
                match = re.search(
                    r"_(\d+)\.pt$",
                    args.resume,
                )

                if match is None:
                    parser.error(
                        "--start-episode is required when the "
                        "checkpoint filename has no episode number."
                    )

                start_episode = int(match.group(1))

            # No saved epsilon, so rebuild it from the current
            # schedule, which may not be the one these weights
            # trained under.
            epsilon = max(
                cfg["eps_end"],
                cfg["eps_start"] * cfg["eps_decay"] ** start_episode,
            )

            # The policy fills the empty replay buffer with
            # epsilon-greedy actions instead of random ones.
            random_warmup = False

            print(
                "Weight-only checkpoint: the optimizer, target "
                "network and replay buffer start fresh, and epsilon "
                "follows the current schedule."
            )

        if start_episode >= cfg["max_episodes"]:
            parser.error(
                f"{args.resume} has already completed "
                f"{start_episode} episodes; raise max_episodes "
                f"({cfg['max_episodes']}) to train further."
            )

        print(
            f"Resumed from {args.resume} "
            f"after episode {start_episode}"
        )

    # --------------------------------------------------------------
    # Environment
    # --------------------------------------------------------------

    env = TurtleBotEnv(
        max_episode_steps=cfg["max_steps"],
    )

    # Spaces keep their own RNG, which set_seed() does not reach.
    env.action_space.seed(cfg["seed"])

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
    # Logging / checkpoints
    # --------------------------------------------------------------

    run_name = time.strftime("%Y%m%d-%H%M%S")

    if args.resume is not None:
        run_name += f"_from{start_episode}"

    writer = SummaryWriter(
        log_dir=f"runs/dqn_turtlebot/{run_name}"
    )

    os.makedirs("checkpoints", exist_ok=True)

    # --------------------------------------------------------------
    # Training state
    # --------------------------------------------------------------

    # Last episode that has finished, for the interrupt checkpoint.
    completed_episodes = start_episode

    def save_checkpoint(path: str) -> None:
        # Reads the loop's values at call time: the last finished
        # episode, and the epsilon the next episode will use.
        agent.save(
            path,
            episode=completed_episodes,
            epsilon=epsilon,
            total_steps=total_steps,
        )

    try:
        # ----------------------------------------------------------
        # Episodes
        # ----------------------------------------------------------

        for episode in range(
            start_episode + 1,
            cfg["max_episodes"] + 1,
        ):
            state, _ = env.reset()

            done = False
            episode_reward = 0.0
            episode_length = 0
            collided = False
            episode_losses = []
            step_durations = []

            # ------------------------------------------------------
            # Episode loop
            # ------------------------------------------------------

            while not done:

                # Initial experience collection.
                if (
                    random_warmup
                    and total_steps < cfg["learning_starts"]
                ):
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

                step_durations.append(info["step_duration"])

            completed_episodes = episode

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

            # Sim seconds per step. Values above CONTROL_DT mean the
            # simulator is outrunning the agent loop.
            mean_step_duration = (
                sum(step_durations) / len(step_durations)
            )
            max_step_duration = max(step_durations)

            writer.add_scalar(
                "step_duration/mean",
                mean_step_duration,
                episode,
            )

            writer.add_scalar(
                "step_duration/max",
                max_step_duration,
                episode,
            )

            print(
                f"ep {episode:4d} | "
                f"reward {episode_reward:8.2f} | "
                f"survived {episode_length:4d} steps | "
                f"collided {int(collided)} | "
                f"eps {epsilon:.3f} | "
                f"dt {mean_step_duration:.3f}/"
                f"{max_step_duration:.3f}"
            )

            # ------------------------------------------------------
            # Periodic checkpoint
            # ------------------------------------------------------

            if episode % cfg["save_every"] == 0:
                checkpoint_path = (
                    f"checkpoints/"
                    f"dqn_turtlebot_{episode}.pt"
                )

                save_checkpoint(checkpoint_path)

                print(
                    f"Saved checkpoint: "
                    f"{checkpoint_path}"
                )

        # ----------------------------------------------------------
        # Final checkpoint
        # ----------------------------------------------------------

        # Named by episode like the periodic ones, not "final": the
        # last checkpoint is not necessarily the best. rl.evaluate
        # ranks them.
        final_checkpoint = (
            f"checkpoints/"
            f"dqn_turtlebot_{completed_episodes}.pt"
        )

        save_checkpoint(final_checkpoint)

        print(
            f"\nTraining complete.\n"
            f"Final checkpoint: {final_checkpoint}"
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

        # Name it after the last finished episode so it can be
        # resumed with --resume directly.
        interrupted_checkpoint = (
            f"checkpoints/"
            f"dqn_turtlebot_interrupted_{completed_episodes}.pt"
        )

        # Ignore further Ctrl-C until the checkpoint is on disk.
        previous_handler = signal.signal(
            signal.SIGINT,
            signal.SIG_IGN,
        )

        try:
            save_checkpoint(interrupted_checkpoint)
        finally:
            signal.signal(signal.SIGINT, previous_handler)

        print(
            f"Saved interrupted checkpoint: "
            f"{interrupted_checkpoint}"
        )

    except Exception:
        # A crash (e.g. a sensor timeout) would otherwise lose every
        # episode since the last periodic checkpoint. Save, then let the
        # error propagate so it is still reported.
        crashed_checkpoint = (
            f"checkpoints/"
            f"dqn_turtlebot_crashed_{completed_episodes}.pt"
        )

        save_checkpoint(crashed_checkpoint)

        print(
            f"\nTraining crashed. Saved checkpoint: "
            f"{crashed_checkpoint}"
        )

        raise

    finally:
        env.close()
        writer.close()


if __name__ == "__main__":
    main()
