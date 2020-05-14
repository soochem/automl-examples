# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# reference
# https://nni.readthedocs.io/en/latest/NAS/NasGuide.html#extend-the-ability-of-one-shot-trainers
# https://github.com/pytorch/examples/blob/master/mnist/
# https://github.com/microsoft/nni/tree/bf1c79a6ad9a89e6c122f4d374c3593be7b09ee8/examples/nas/enas


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from nni.nas.pytorch import mutables, enas
from datetime import datetime


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=28, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(28, 56, 3, 1)
        self.conv3 = nn.Conv2d(56, 56, 1, 1)
        # declaring that there is exactly one candidate to choose from
        # search strategy will choose one or None
        self.skipcon = mutables.InputChoice(n_candidates=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(56*13*13, 128)  #64*12*12
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x0 = self.skipcon([x])  # choose one or none from [x]
        x = self.conv3(x)
        if x0 is not None:  # skipconnection is open
            x += x0
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def reward_accuracy(output, target, topk=(1,)):
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return (predicted == target).sum().item() / batch_size


# this is exactly same as traditional model training
model = Net()
# MEAN = [0.49139968, 0.48215827, 0.44653124]
# STD = [0.24703233, 0.24348505, 0.26158768]
transf = [
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip()
]
normalize = [
    transforms.ToTensor(),
    # transforms.Normalize(MEAN, STD)
]
dataset_train = datasets.FashionMNIST(root="./s",
                                      train=True,
                                      download=True,
                                      transform=transforms.Compose(transf + normalize),
                                      )
dataset_valid = datasets.FashionMNIST(root="./s",
                                      train=False,
                                      download=True,
                                      transform=transforms.Compose(normalize),
                                      )

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)


# use NAS here
def top1_accuracy(output, target):
    # this is the function that computes the reward, as required by ENAS algorithm
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return (predicted == target).sum().item() / batch_size


def metrics_fn(output, target):
    # metrics function receives output and target and computes a dict of metrics
    return {"acc1": reward_accuracy(output, target)}


trainer = enas.EnasTrainer(model,
                           loss=criterion,
                           metrics=metrics_fn,
                           reward_function=top1_accuracy,
                           optimizer=optimizer,
                           batch_size=128,
                           num_epochs=1,  # 10 epochs
                           dataset_train=dataset_train,
                           dataset_valid=dataset_valid,
                           log_frequency=10)  # print log every 10 steps

trainer.train()  # training
model_id = datetime.today().strftime("%Y-%m-%d_%H%M%S")
model_dir = "../../../../data"
trainer.export(file=model_dir+"/final_architecture"+model_id + ".json")  # export the final architecture to file
