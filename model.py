import tensorflow as tf
from keras.layers import BatchNormalization, MaxPooling2D
from keras.models import load_model, Sequential
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense


def create_model():
    dropout = 0.0

    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=(80,80,1)))
    model.add(MaxPooling2D(pool_size=3))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=3))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())

    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=3))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='mse')

    return model

def create_target_model():
    dropout = 0.0

    target_model = Sequential()

    target_model.add(Conv2D(64, kernel_size=3, input_shape=(80, 80, 1), padding='same', activation='relu', batch_size=64))
    target_model.add(MaxPooling2D(pool_size=2))
    target_model.add(Dropout(dropout))
    target_model.add(BatchNormalization())

    target_model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    target_model.add(MaxPooling2D(pool_size=2))
    target_model.add(Dropout(dropout))
    target_model.add(BatchNormalization())

    target_model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    target_model.add(MaxPooling2D(pool_size=2))
    target_model.add(Dropout(dropout))
    target_model.add(BatchNormalization())

    target_model.add(Flatten())
    target_model.add(Dense(512, activation='relu'))
    target_model.add(Dense(256, activation='relu'))
    target_model.add(Dense(3, activation='softmax'))

    target_model.compile(optimizer='adam', loss='mse')

    return target_model

