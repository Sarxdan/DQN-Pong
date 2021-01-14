import tensorflow as tf
from keras.layers import BatchNormalization, MaxPooling2D
from keras.models import load_model, Sequential
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense

import numpy as np

import random

# 0 -> No move
# 2 -> Up
# 3 -> Down
actions = [0, 2, 3]


class Model():
    def __init__(self):
        dropout = 0.0

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(80, 80, 1)))
        model.add(MaxPooling2D(pool_size=3))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())

        model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=3))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())

        model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=3))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer='adam', loss='mse')

        self.model = model

        self.checkpoint = tf.keras.callbacks.ModelCheckpoint("ModelWeights.hdf5", monitor='loss', verbose=1,
                                                             save_best_only=True)

        # Statics
        self.gamma = 0.85

        self.epsilon = 0.99
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.00001

    # Load from either given model or previous file
    def load_weights(self, weights, target_model=True):
        if target_model:
            self.model.set_weights(weights.get_weights)
        else:
            self.model.load_weights("ModelWeights.hdf5")

    def step(self, env, state):
        # Epsilon-greedy algorithm
        if random.uniform(0, 1) <= self.epsilon:
            action = actions[np.random.randint(0, 3)]
        else:
            state = np.reshape(state, (1, 80, 80, 1))
            action = actions[np.argmax(self.model.predict(state))]

        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)

        state, reward, done, info = env.step(action)
        return state, action, reward, done, info

    def train(self, target_model, replay_memory):
        if len(replay_memory.replay_memory) >= replay_memory.batch_size:

            experiences = replay_memory.sample_experiences()

            for state, action, reward, next_state, done in experiences:
                target = target_model.predict(state)
                if done:
                    target[0][action] = reward
                else:
                    # Reshape to array shaped for batch
                    next_state = np.reshape(next_state, (1, 80, 80, 1))
                    q_future = np.amax(target_model.predict(next_state)[0])
                    target[0][action] = reward + self.gamma * q_future
                self.model.fit(state, target, epochs=1, verbose=0)

    def validate(self, env, state):
        state = np.reshape(state, (1, 80, 80, 1))
        action = actions[np.argmax(self.model.predict(state))]
        state, reward, done, info = env.step(action)
        return state, action, reward, done, info
