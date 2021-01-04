# Using Open AI Gym environment
# Actions are 0-6 where as 2 and 3 are up and down
# State is a 210x160 image with RGB channels


import tensorflow as tf
import gym

import numpy as np
import matplotlib.pyplot as plt

import time

up = 2
down = 3


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


def main():
    unlimited_refresh = False
    refresh_rate = 1 / 60
    time_stamp = time.time()

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
    # Inspect a pre-processed image
    first_obs = process_image(first_obs).reshape(80, 80)
    plt.imshow(first_obs, cmap='gray')
    plt.show()

    while True:
        if unlimited_refresh or time.time() - time_stamp >= refresh_rate:
            time_stamp = time.time()

            env.render()
            state, reward, done, info = env.step(env.action_space.sample())

            # TODO: Add x previous frames
            state = process_image(state).reshape(80, 80)

            if done:
                env.reset()

            # TODO: Create a model

            # TODO: Create a target-model for predicting the target (future reward)

            # TODO: Setup replay memory

            # TODO: Implement the q-learning part

            # TODO: Train the model

            # TODO: Display the history of training

    env.close()


main()
