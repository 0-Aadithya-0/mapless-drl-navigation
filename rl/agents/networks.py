"""Neural network definitions.

QNetwork is unchanged from CartPole — only the numbers you pass in differ:
CartPole was QNetwork(4, 2); the robot's DQN is QNetwork(24, 4).
"""
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Reads a state, outputs one Q-value (score) per action."""

    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),   # one score per action
        )

    def forward(self, state):
        return self.net(state)
