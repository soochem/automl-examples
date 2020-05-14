# Model: LeNet
# Dataset: FashionMNIST
# Framework : TensorFlow

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, Activation
import numpy as np
import argparse
from datetime import datetime, timezone


class Net:
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


def train(args, model, x_train, y_train, x_test, y_test):
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=args.learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'],
                  )

    print("Training...")
    katib_metric_log_callback = KatibMetricLog()
    training_history = model.fit(x_train, y_train,
                                 batch_size=64,
                                 epochs=5,
                                 validation_data=(x_test, y_test),
                                 callbacks=[katib_metric_log_callback],
                                 )
    print("\ntraining_history:", training_history.history)


def test(model, x_test, y_test):
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    print('test loss, test acc:', results)


def build_model():
    # parsing args
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--dropout', default=0.2, type=float)
    args = parser.parse_args()

    # Load data
    mnist = tf.keras.datasets.fashion_mnist  # todo : try to load file
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Use smaller data
    x_train = x_train[:10000]
    y_train = y_train[:10000]

    # Expand dimension
    x_train = x_train[:, :, :, np.newaxis]  # add additional channel
    x_test = x_test[:, :, :, np.newaxis]

    # Train model
    input_shape = (28, 28, 1)
    model = Net.build(args, input_shape)
    train(args, model, x_train, y_train, x_test, y_test)
    test(model, x_test, y_test)


class KatibMetricLog(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # RFC 3339
        local_time = datetime.now(timezone.utc).astimezone().isoformat()
        print("\nEpoch {}".format(epoch+1))
        print("{} Train-accuracy={:.4f}".format(local_time, logs['acc']))
        print("{} loss={:.4f}".format(local_time, logs['loss']))
        print("{} Validation-accuracy={:.4f}".format(local_time, logs['val_acc']))
        print("{} Validation-loss={:.4f}".format(local_time, logs['val_loss']))


if __name__ == '__main__':
    build_model()
