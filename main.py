# Using Open AI Gym environment
# Actions are 0-6 where as 2 and 3 are up and down
# State is a 210x160 image with RGB channels


import tensorflow as tf
from keras.layers import BatchNormalization, MaxPooling2D
from keras.models import load_model, Sequential
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint

import gym

import numpy as np
import matplotlib.pyplot as plt

import time
import random

from replay_memory import *
from model import *

from tensorflow.python.client import device_lib


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


def main():
    validation_mode = False

    # ---Setup variables---
    unlimited_refresh = True
    refresh_rate = 1 / 60
    time_stamp = time.time()

    episodes = 300

    replay_memory = ReplayMemory(5000, 32)

    rewards = np.array(episodes)

    # Check for GPU
    if tf.test.gpu_device_name():
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

    else:
        print("Please install GPU version of TF")

    # Setup env and start rendering
    env = gym.make("Pong-v0")
    print("Action space: " + str(env.action_space))
    print("Observation space: " + str(env.observation_space))

    current_state = env.reset()
    # Inspect the first frame
    current_state = process_image(current_state, replay_memory.batch_size).reshape(80, 80)
    plt.imshow(current_state, cmap='gray')
    plt.show()

    model = Model()
    target_model = Model()

    if validation_mode is False:
        for _ in range(episodes):
            print("Current episode: " + str(_))
            current_state = process_image(env.reset(), replay_memory.batch_size)

            # Wait until the ball has spawn
            for pre in range(25):
                current_state, reward, done, info = env.step(0)
            current_state = process_image(current_state, replay_memory.batch_size)

            steps = 0

            while True:
                if unlimited_refresh or time.time() - time_stamp >= refresh_rate:
                    time_stamp = time.time()
                    env.render()

                    new_state, action, reward, done, info = model.step(env, current_state)
                    new_state = process_image(new_state, replay_memory.batch_size)  # Single state

                    # Add experience to replay memory
                    replay_memory.add_experiences(current_state, action, reward, new_state, done)
                    current_state = new_state

                    model.train(target_model.model, replay_memory)

                    steps = steps + 1

                    if steps % 20 == 0:
                        target_model.model.set_weights(model.model.get_weights())

                    if done:
                        # TODO: Display the history of training
                        rewards = np.append(rewards, reward)
                        break


    elif validation_mode:
        for _ in range(episodes):
            current_state = process_image(env.reset(), replay_memory.batch_size)

            # Wait until the ball has spawn
            for pre in range(25):
                current_state, reward, done, info = env.step(0)
            current_state = process_image(current_state, replay_memory.batch_size)

            steps = 0

            while True:
                if unlimited_refresh or time.time() - time_stamp >= refresh_rate:
                    time_stamp = time.time()
                    env.render()

                    new_state, action, reward, done, info = model.validate(env, current_state)
                    new_state = process_image(new_state, replay_memory.batch_size)  # Single state

                    # Add experience to replay memory
                    replay_memory.add_experiences(current_state, action, reward, new_state, done)
                    current_state = new_state

                    model.train(target_model, replay_memory)

                    steps = steps + 1

                    if done:
                        # TODO: Display the history of training
                        rewards = np.append(rewards, reward)
                        break

    env.close()


main()
