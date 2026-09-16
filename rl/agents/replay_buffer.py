"""Experience replay buffer: stores past experiences, returns random batches.

Unchanged from the CartPole version — it stores and samples arrays and does not
care whether a state is 4 numbers (CartPole) or 24 (the robot).
"""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)
        self.index = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        i = self.index
        self.states[i]      = state
        self.actions[i]     = action
        self.rewards[i]     = reward
        self.next_states[i] = next_state
        self.dones[i]       = done
        self.index = (self.index + 1) % self.capacity
        self.size  = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idxs], self.actions[idxs], self.rewards[idxs],
                self.next_states[idxs], self.dones[idxs])

    def __len__(self):
        return self.size
