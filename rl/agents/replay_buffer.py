"""Experience replay buffer: stores past experiences, returns random batches.

Storage and sampling are unchanged from the CartPole version — it does not
care whether a state is 4 numbers (CartPole) or 24 (the robot).
state_dict/load_state_dict let a checkpoint carry the buffer, so a resumed
run keeps its experience.
"""
import numpy as np
import torch


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

    def state_dict(self):
        # torch tensors, not numpy arrays: torch.load's default
        # weights_only=True refuses to unpickle numpy
        n = self.size
        return {
            "states":      torch.tensor(self.states[:n]),
            "actions":     torch.tensor(self.actions[:n]),
            "rewards":     torch.tensor(self.rewards[:n]),
            "next_states": torch.tensor(self.next_states[:n]),
            "dones":       torch.tensor(self.dones[:n]),
            "index": self.index,
        }

    def load_state_dict(self, state):
        n = len(state["states"])
        if n > self.capacity:
            raise ValueError(f"checkpoint buffer holds {n} transitions, "
                             f"but capacity is {self.capacity}")
        self.states[:n]      = state["states"].numpy()
        self.actions[:n]     = state["actions"].numpy()
        self.rewards[:n]     = state["rewards"].numpy()
        self.next_states[:n] = state["next_states"].numpy()
        self.dones[:n]       = state["dones"].numpy()
        self.size = n
        # a full buffer continues its ring; a partial one appends after the data
        self.index = state["index"] if n == self.capacity else n
