import random
from collections import deque


class ReplayMemory:

    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)

    def add(self, experience_tuple):
        self.buffer.append(experience_tuple)

    def get_size(self):
        return len(self.buffer)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)

        if batch_size > buffer_size:
            return random.sample(self.buffer, buffer_size)
        else:
            return random.sample(self.buffer, batch_size)

        # indices = np.random.choice(np.arange(buffer_size),
        #                          size = batch_size,
        #                          replace = False)
        #
        # return [self.buffer[i] for i in indices]


