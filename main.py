# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import tensorflow as tf
import gym
import time

def main():
    unlimitedRefresh = False
    refreshRate = 1/60
    timeStamp = time.time()

    # Check for GPU
    if tf.test.gpu_device_name():
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

    else:
        print("Please install GPU version of TF")

    # Setup env and start rendering
    env = gym.make("Pong-v0")
    env.reset()

    while(True):
        if(unlimitedRefresh or time.time() - timeStamp >= refreshRate):
            timeStamp = time.time()

            env.render()
            env.step(env.action_space.sample())

    env.close()


main()