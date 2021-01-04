from collections import deque

class replay_memory:
    def __init__(self, bufferSize, miniBatchSize):
        self.replayBuffer = deque(maxlen=bufferSize)
        self.miniBatchSize = miniBatchSize

    def add_experiences(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.replayBuffer.append(experience)

    def sample_experiences(self):
        # TODO: Pick random experiences
        return