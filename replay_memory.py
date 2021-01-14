from collections import deque
import numpy as np
import random

class ReplayMemory:
    def __init__(self, buffer_size, batch_size):
        self.replay_memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def add_experiences(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def sample_experiences(self):
        # TODO: Pick random experiences
        #indices = np.random.randint(len(self.replay_memory), size=self.batch_size)
        #batch = [self.replay_memory[index] for index in indices]
        #states, actions, rewards, next_states, dones = [
        #    np.array([experience[field_index] for experience in batch])
        #    for field_index in range(5)]
        batch = random.sample(self.replay_memory, self.batch_size)
        return batch
