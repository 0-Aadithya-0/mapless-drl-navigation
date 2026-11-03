"""The DQN agent: the brain (Q-network) + a slow target copy + memory + learning.

Identical in structure to the CartPole version you built and understand. The only
differences for the robot are the numbers you pass in (state_dim=24, n_actions=4),
which come from the config file.
"""
import os
import random
import torch
import torch.nn.functional as F

from rl.agents.networks import QNetwork
from rl.agents.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_dim, n_actions,
                 gamma=0.99, lr=5e-4, buffer_size=50000,
                 batch_size=64, target_update_every=500, device="cpu"):
        self.state_dim = state_dim
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

    def warmup(self):
        # the first GPU call of each kernel is slow (~0.5 s: lazy kernel
        # loading, cuBLAS setup) and the sim keeps running meanwhile, so that
        # step would last seconds; run select_action and learn once on a
        # throwaway agent first — loaded kernels stay loaded for the process
        scratch = DQNAgent(self.state_dim, self.n_actions,
                           buffer_size=self.batch_size,
                           batch_size=self.batch_size, device=self.device)
        state = [0.0] * self.state_dim
        for _ in range(self.batch_size):
            scratch.buffer.add(state, 0, 0.0, state, 0.0)
        scratch.select_action(state, epsilon=0.0)
        scratch.learn()

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

    def save(self, path, **training_state):
        # everything that changes during training, so a resume continues
        # the same run; training_state carries the loop's own counters
        checkpoint = {
            "q_net":       self.q_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "learn_steps": self.learn_steps,
            "buffer":      self.buffer.state_dict(),
            **training_state,
        }
        # write then rename: an interrupted save never leaves a truncated file
        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)

    def load(self, path):
        # returns the training_state saved with the checkpoint, or None for
        # old checkpoints that hold only Q-network weights
        checkpoint = torch.load(path, map_location="cpu")
        if "q_net" not in checkpoint:
            self.q_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(self.q_net.state_dict())
            return None
        self.q_net.load_state_dict(checkpoint.pop("q_net"))
        self.target_net.load_state_dict(checkpoint.pop("target_net"))
        self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        self.learn_steps = checkpoint.pop("learn_steps")
        self.buffer.load_state_dict(checkpoint.pop("buffer"))
        return checkpoint
