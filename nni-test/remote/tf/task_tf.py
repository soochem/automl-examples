# Model: LeNet
# Dataset: FashionMNIST
# Framework : TensorFlow
# From Katib Example

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import argparse
import model_tf


def train(args, model, x_train, y_train, x_test, y_test):
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=args.learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'],
                  )

    print("Training...")
    training_history = model.fit(x_train, y_train,
                                 batch_size=64,
                                 epochs=5,
                                 validation_data=(x_test, y_test),
                                 )
    print("\ntraining_history:", training_history.history)


def test(model, x_test, y_test):
    print('\n# Evaluate on test s')
    results = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    print('test loss, test acc:', results)


def build_model():
    # parsing args
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--dropout', default=0.2, type=float)
    args = parser.parse_args()

    # Load s
    mnist = tf.keras.datasets.fashion_mnist  # todo : try to load file
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Use smaller s
    x_train = x_train[:10000]
    y_train = y_train[:10000]

    # Expand dimension
    x_train = x_train[:, :, :, np.newaxis]  # add additional channel
    x_test = x_test[:, :, :, np.newaxis]

    # Train model
    input_shape = (28, 28, 1)
    model = model_tf.LeNet.build(args, input_shape)
    train(args, model, x_train, y_train, x_test, y_test)
    test(model, x_test, y_test)


if __name__ == '__main__':
    build_model()
