"""The DQN agent: the brain (Q-network) + a slow target copy + memory + learning.

Identical in structure to the CartPole version you built and understand. The only
differences for the robot are the numbers you pass in (state_dim=24, n_actions=4),
which come from the config file.
"""
import random
import torch
import torch.nn.functional as F

from rl.agents.networks import QNetwork
from rl.agents.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_dim, n_actions,
                 gamma=0.99, lr=5e-4, buffer_size=50000,
                 batch_size=64, target_update_every=500, device="cpu"):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.device = device

        # main network (trained) + target network (slow copy)
        self.q_net      = QNetwork(state_dim, n_actions).to(device)
        self.target_net = QNetwork(state_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())   # start identical

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size, state_dim)
        self.learn_steps = 0

    @torch.no_grad()
    def select_action(self, state, epsilon):
        # epsilon-greedy: explore with prob epsilon, else pick the best action
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.q_net(state_t).argmax(dim=1).item())

    def learn(self):
        # wait until there are enough experiences to form a batch
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states      = torch.as_tensor(states,      dtype=torch.float32, device=self.device)
        actions     = torch.as_tensor(actions,     dtype=torch.int64,   device=self.device)
        rewards     = torch.as_tensor(rewards,     dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones       = torch.as_tensor(dones,       dtype=torch.float32, device=self.device)

        # what the network currently thinks about the action we actually took
        q_pred = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # the Bellman target, using the FROZEN target network
        with torch.no_grad():
            q_next = self.target_net(next_states).max(dim=1).values
            target = rewards + self.gamma * q_next * (1.0 - dones)

        loss = F.smooth_l1_loss(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # occasionally sync the target network
        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
