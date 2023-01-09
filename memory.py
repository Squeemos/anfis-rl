import random
from collections import deque

class Memory(object):
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)

    def store(self, experience):
        self.memory.append(experience)

    def sample(self, n_samples):
        samples = random.sample(self.memory, n_samples)
        return zip(*samples)

    def __len__(self):
        return len(self.memory)
