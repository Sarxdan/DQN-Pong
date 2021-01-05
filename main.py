# Using Open AI Gym environment
# Actions are 0-6 where as 2 and 3 are up and down
# State is a 210x160 image with RGB channels


import tensorflow as tf
import gym

import numpy as np
import matplotlib.pyplot as plt

import time
import random

from replay_memory import*

# 0 -> No move
# 2 -> Up
# 3 -> Down
actions = [0, 2, 3]

# TODO: Pre-process the image

def process_image(state):
    # Pre-process 210x160x3 frame into 6400(80x80) 1D float vector
    state = state[35:195]  # Crop
    state = state[::2, ::2, 0]  # Downsample by factor of 2
    # Set everything to black and white
    state[state == 144] = 0
    state[state == 109] = 0
    state[state != 0] = 1
    return state.astype(np.float).ravel()


def step(env, epsilon):
    action = 0

    # Epsilon-greedy algorithm
    if random.uniform(0, 1) <= epsilon:
        action = actions[np.random.randint(0, 3)]
    else:
        action = actions[2]         # TODO: Choose from model prediction

    state, reward, done, info = env.step(action)
    return state, action, reward, done, info


def main():
    # ---Setup variables---
    unlimited_refresh = True
    refresh_rate = 1 / 60
    time_stamp = time.time()

    episodes = 200

    epsilon = 0.99
    min_epsilon = 0.1
    epsilon_decay = 0.000

    replay_memory = ReplayMemory(5000, 64)

    # Check for GPU
    if tf.test.gpu_device_name():
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

    else:
        print("Please install GPU version of TF")

    # Setup env and start rendering
    env = gym.make("Pong-v0")
    print("Action space: " + str(env.action_space))
    print("Observation space: " + str(env.observation_space))

    first_obs = env.reset()
    # Inspect the first frame
    first_obs = process_image(first_obs).reshape(80, 80)
    plt.imshow(first_obs, cmap='gray')
    plt.show()

    for _ in range(episodes):
        first_obs = env.reset()

        # Wait until the ball has spawn
        for pre in range(20):
            env.step(0)

        first_obs = process_image(first_obs)

        current_state = first_obs

        while True:
            if unlimited_refresh or time.time() - time_stamp >= refresh_rate:
                time_stamp = time.time()
                env.render()

                new_state, action, reward, done, info = step(env, epsilon)

                epsilon = max(min_epsilon, epsilon - epsilon_decay)

                # TODO: Add x previous frames
                new_state = process_image(new_state)

                # TODO: Create a model

                # TODO: Create a target-model for predicting the target (future reward)

                # TODO: Setup replay memory

                replay_memory.add_experiences(current_state, action, reward, done, new_state)
                current_state = new_state

                # TODO: Implement the q-learning part

                # TODO: Train the model

                # TODO: Display the history of training

                if done:
                    break

    env.close()


main()
