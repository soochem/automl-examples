# Model: Lenet
# Dataset: MNIST
# Framework: MXNet
# from Katib Example
# https://github.com/kubeflow/katib/tree/master/examples/v1alpha3/mxnet-mnist
# from AutoGloun Example
# https://github.com/awslabs/autogluon/blob/00cce171aa92d3bff4cec73214355c01b0bc19d3/autogluon/task/image_classification/nets.py


import logging
import mxnet as mx
from mxnet import gluon, init
from mxnet.gluon import nn

from autogluon.model_zoo import get_model
from autogluon.core import *

logger = logging.getLogger(__name__)


class ConvReLU(mx.gluon.HybridBlock):
    def __init__(self, in_channels, channels, kernel, stride):
        super().__init__()
        padding = (kernel - 1) // 2
        self.conv = nn.Conv2D(channels, kernel, stride, padding, in_channels=in_channels)
        self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        return self.relu(self.bn(self.conv(x)))


# class ConvBNReLU(mx.gluon.HybridBlock):
#     def __init__(self, in_channels, channels, kernel, stride):
#         super().__init__()
#         padding = (kernel - 1) // 2
#         self.conv = nn.Conv2D(channels, kernel, stride, padding, in_channels=in_channels)
#         self.bn = nn.BatchNorm(in_channels=channels)
#         self.relu = nn.Activation('relu')
#
#     def hybrid_forward(self, F, x):
#         return self.relu(self.bn(self.conv(x)))
#
#
# class ResUnit(mx.gluon.HybridBlock):
#     def __init__(self, in_channels, channels, hidden_channels, kernel, stride):
#         super().__init__()
#         self.conv1 = ConvBNReLU(in_channels, hidden_channels, kernel, stride)
#         self.conv2 = ConvBNReLU(hidden_channels, channels, kernel, 1)
#         if in_channels == channels and stride == 1:
#             self.shortcut = Identity()
#         else:
#             self.shortcut = nn.Conv2D(channels, 1, stride, in_channels=in_channels)
#
#     def hybrid_forward(self, F, x):
#         return self.conv2(self.conv1(x)) + self.shortcut(x)


# def mnist_net():
#     mnist_net = gluon.nn.Sequential()
#     mnist_net.add(ResUnit(1, 8, hidden_channels=8, kernel=3, stride=2))
#     mnist_net.add(ResUnit(8, 8, hidden_channels=8, kernel=5, stride=2))
#     mnist_net.add(ResUnit(8, 16, hidden_channels=8, kernel=3, stride=2))
#     mnist_net.add(nn.GlobalAvgPool2D())
#     mnist_net.add(nn.Flatten())
#     mnist_net.add(nn.Activation('relu'))
#     mnist_net.add(nn.Dense(10, in_units=16))
#     return mnist_net


class LeNet:
    @staticmethod
    def build():
        model = nn.Sequential()
        model.add(ConvReLU(1, 20, 5, 1))
        model.add(nn.MaxPool2D())
        model.add(ConvReLU(20, 50, 5, 1))
        model.add(nn.MaxPool2D())
        model.add(nn.Flatten())
        model.add(nn.Dense(500))
        model.add(nn.Activation("relu"))
        # model.add(Dropout(args.dropout))
        model.add(nn.Dense(10,  in_units=16))
        model.add(nn.Activation("softmax"))
        return model


