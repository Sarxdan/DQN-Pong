# Using Open AI Gym environment
# Actions are 0-6 where as 2 and 3 are up and down
# State is a 210x160 image with RGB channels


import tensorflow as tf
from keras.layers import BatchNormalization, MaxPooling2D
from keras.models import load_model, Sequential
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense

import gym

import numpy as np
import matplotlib.pyplot as plt

import time
import random

from replay_memory import *
from model import *

from tensorflow.python.client import device_lib

# 0 -> No move
# 2 -> Up
# 3 -> Down
actions = [0, 2, 3]


# TODO: Pre-process the image

def process_image(state, batch_size):
    # Pre-process 210x160x3 frame into 6400(80x80) 1D float vector
    state = state[35:195]  # Crop
    state = state[::2, ::2, 0]  # Downsample by factor of 2
    # Set everything to black and white
    state[state == 144] = 0
    state[state == 109] = 0
    state[state != 0] = 1
    # return state.astype(np.float).ravel()

    return np.reshape(state.astype(np.float), (80, 80, 1))


def step(env, epsilon):
    # Epsilon-greedy algorithm
    if random.uniform(0, 1) <= epsilon:
        action = actions[np.random.randint(0, 3)]
    else:
        action = actions[2]  # TODO: Choose from model prediction

    state, reward, done, info = env.step(action)
    return state, action, reward, done, info


def train(model, target_model, replay_memory, gamma):
    if len(replay_memory.replay_memory) >= replay_memory.batch_size:
        experiences = replay_memory.sample_experiences()

        states, actions, rewards, next_states, dones = replay_memory.sample_experiences()
        targets = np.empty(replay_memory.batch_size)

        # Reshape states to 1 dimension
        #states = np.reshape(states, (replay_memory.batch_size, 80, 80, 1))
        #next_states = np.reshape(next_states, (replay_memory.batch_size, 80, 80, 1))

        for i in range(replay_memory.batch_size):
            if dones[i]:
                target = rewards[i]
            else:
                # Reshape to array shaped for batch
                state = np.reshape(next_states[i], (1, 80, 80, 1))
                q_future = np.amax(target_model.predict(state))
                target = rewards[i] + gamma * q_future
            targets[i] = target
        model.fit(states, targets)

        # TODO: Implement the q-learning part
        # for experience in experiences:
        #    state, action, reward, next_state, done = experience
        #    target = target_model.predict(state)
        #    if done:
        #        target[0][action] = reward
        #    else:
        #        q_future = max(target_model.predict(new_state)[0])
        #        target[0][action] = reward + q_future * gamma

        # TODO: Train the model
        #    model.fit(state, target)


def main():
    # ---Setup variables---
    unlimited_refresh = True
    refresh_rate = 1 / 60
    time_stamp = time.time()

    episodes = 300

    epsilon = 0.99
    min_epsilon = 0.1
    epsilon_decay = 0.00001

    gamma = 0.85

    replay_memory = ReplayMemory(5000, 1)

    rewards = np.array(episodes)

    # Check for GPU
    if tf.test.gpu_device_name():
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

    else:
        print("Please install GPU version of TF")

    print(device_lib.list_local_devices())

    # Setup env and start rendering
    env = gym.make("Pong-v0")
    print("Action space: " + str(env.action_space))
    print("Observation space: " + str(env.observation_space))

    current_state = env.reset()
    # Inspect the first frame
    current_state = process_image(current_state, replay_memory.batch_size).reshape(80, 80)
    plt.imshow(current_state, cmap='gray')
    plt.show()

    # TODO: Create a model
    # TODO: Create a target-model for predicting the target (future reward)
    model = create_model()
    target_model = create_model()

    for _ in range(episodes):
        current_state = process_image(env.reset(), replay_memory.batch_size)
        # current_state = np.array([first_obs, first_obs, first_obs])

        # Wait until the ball has spawn
        for pre in range(25):
            current_state, reward, done, info = env.step(0)
            # np.append(current_state[1:3], [process_image(observation)], axis=0)
        current_state = process_image(current_state, replay_memory.batch_size)

        while True:
            if unlimited_refresh or time.time() - time_stamp >= refresh_rate:
                time_stamp = time.time()
                env.render()

                new_state, action, reward, done, info = step(env, epsilon)

                # Decrease epsilon
                epsilon = max(min_epsilon, epsilon - epsilon_decay)

                # TODO: Add x previous frames
                new_state = process_image(new_state, replay_memory.batch_size)  # Single state
                # new_state = np.append(current_state[1:3], [new_state], axis=0)    # Stack states

                # TODO: Setup replay memory
                # Add experience to replay memory
                replay_memory.add_experiences(current_state, action, reward, new_state, done)
                current_state = new_state

                train(model, target_model, replay_memory, gamma)

                if done:
                    # TODO: Display the history of training
                    rewards = np.append(rewards, reward)
                    break

    env.close()


main()
