"""Evaluate a trained DQN with exploration turned OFF.

Reports the numbers your Week 2 done-check asks for: average reward,
average survival time, and collision rate.

Run from the repo root:
    python -m rl.evaluate --env turtlebot --checkpoint checkpoints/dqn_turtlebot_final.pt
"""
import argparse

import numpy as np
import torch

from rl.utils.config import load_config
from rl.agents.dqn import DQNAgent
from rl.train import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["turtlebot", "cartpole"], default="turtlebot")
    p.add_argument("--config", default="rl/configs/dqn.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = make_env(args.env, cfg["max_steps"])
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(state_dim, n_actions, device=device)
    agent.load(args.checkpoint)

    rewards, lengths, collisions = [], [], 0

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        done = False
        ep_reward, ep_len = 0.0, 0
        info = {}
        while not done:
            action = agent.select_action(state, epsilon=0.0)   # greedy: no exploration
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_len += 1
        rewards.append(ep_reward)
        lengths.append(ep_len)
        if info.get("collision"):
            collisions += 1
        print(f"ep {ep:3d} | reward {ep_reward:8.1f} | survived {ep_len:4d} steps")

    n = args.episodes
    print("\n==== Evaluation ====")
    print(f"episodes:        {n}")
    print(f"avg reward:      {np.mean(rewards):.1f}")
    print(f"avg survival:    {np.mean(lengths):.1f} steps")
    print(f"collision rate:  {collisions / n:.0%}")

    env.close()


if __name__ == "__main__":
    main()
