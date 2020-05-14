# Model: LeNet
# Dataset: FashionMNIST
# Framework : TensorFlow
# From Katib Example

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, Activation
import numpy as np
import argparse
from datetime import datetime, timezone


class LeNet:
    @staticmethod
    def build(args, input_shape):  # class, input_size at the first line?
        model = Sequential()
        model.add(Conv2D(20, kernel_size=5, padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(50, kernel_size=5, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(Dropout(args.dropout))
        model.add(Dense(10))
        model.add(Activation("softmax"))
        return model
