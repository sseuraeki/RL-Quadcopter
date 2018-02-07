"""Replay Buffer."""

import random

from collections import namedtuple

Experience = namedtuple("Experience",
    field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Fixed-size circular buffer to store experience tuples."""

    def __init__(self, size=1000):
        """Initialize a ReplayBuffer object."""
        self.size = size  # maximum size of buffer
        self.memory = []  # internal memory (list)
        self.idx = 0  # current index into circular buffer
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # TODO: Create an Experience object, add it to memory
        # Note: If memory is full, start overwriting from the beginning
        e = Experience(state, action, reward, next_state, done)
        if len(self.memory) < self.size:
            self.memory.append(e)
        else:
            self.memory[self.idx] = e
            self.idx = (self.idx + 1) % self.size
    
    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        # TODO: Return a list or tuple of Experience objects sampled from memory
        batch = []
        
        while len(batch) < batch_size:
            s = random.choice(self.memory)
            if s not in batch:
                batch.append(s)
        
        return batch

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)