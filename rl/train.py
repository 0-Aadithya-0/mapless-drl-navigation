"""Train the DQN.

Works on the robot (--env turtlebot) and on CartPole (--env cartpole), so you can
test the agent on CartPole before the robot environment is ready. The agent, buffer,
and learning are identical for both — only the environment changes.

Run from the repo root:
    python -m rl.train --env turtlebot
    python -m rl.train --env cartpole     # quick sanity check, no ROS needed
"""
import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from rl.utils.seeding import set_seed
from rl.utils.config import load_config
from rl.agents.dqn import DQNAgent


def make_env(name, max_steps):
    if name == "cartpole":
        import gymnasium as gym
        return gym.make("CartPole-v1", max_episode_steps=max_steps)
    if name == "turtlebot":
        # imported only when needed, so CartPole runs without ROS installed
        from turtlebot3_rl_nav.env import TurtleBotEnv
        return TurtleBotEnv(max_episode_steps=max_steps)
    raise ValueError(f"unknown env: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["turtlebot", "cartpole"], default="turtlebot")
    p.add_argument("--config", default="rl/configs/dqn.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = make_env(args.env, cfg["max_steps"])
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(
        state_dim, n_actions,
        gamma=cfg["gamma"], lr=cfg["lr"],
        buffer_size=cfg["buffer_size"], batch_size=cfg["batch_size"],
        target_update_every=cfg["target_update_every"], device=device,
    )

    writer = SummaryWriter(f"runs/dqn_{args.env}")
    os.makedirs("checkpoints", exist_ok=True)

    epsilon = cfg["eps_start"]
    total_steps = 0

    for episode in range(1, cfg["max_episodes"] + 1):
        state, _ = env.reset()
        done = False
        ep_reward, ep_len, collided = 0.0, 0, False

        while not done:
            # act randomly until the buffer has warmed up, then epsilon-greedy
            if total_steps < cfg["learning_starts"]:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, epsilon)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # store `terminated` (a real crash), NOT `truncated` (a timeout) as the done flag
            agent.buffer.add(state, action, reward, next_state, float(terminated))
            if total_steps >= cfg["learning_starts"]:
                agent.learn()

            state = next_state
            ep_reward += reward
            ep_len += 1
            total_steps += 1
            if info.get("collision"):
                collided = True

        # shrink exploration each episode
        epsilon = max(cfg["eps_end"], epsilon * cfg["eps_decay"])

        # ---- logging (TensorBoard + console) ----
        writer.add_scalar("reward/episode", ep_reward, episode)
        writer.add_scalar("survival_steps", ep_len, episode)   # survival time
        writer.add_scalar("collision", int(collided), episode)
        writer.add_scalar("epsilon", epsilon, episode)
        print(f"ep {episode:4d} | reward {ep_reward:8.1f} | survived {ep_len:4d} steps | "
              f"collided {int(collided)} | eps {epsilon:.2f}")

        if episode % cfg["save_every"] == 0:
            agent.save(f"checkpoints/dqn_{args.env}_{episode}.pt")

    agent.save(f"checkpoints/dqn_{args.env}_final.pt")
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
