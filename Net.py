import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

class Net(nn.Module):
    def __init__(self, h, w, out):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        self.size1 = [(h-2)//2, (w-2)//2]

        self.conv2 = nn.Conv2d(16, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.size2 = [(self.size1[0]-2)//2, (self.size1[1]-2)//2]

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.size3 = [(self.size2[0]-2)//2, (self.size2[1]-2)//2]

        self.dense1 = nn.Linear(self.size3[0]*self.size3[1]*128, 128)
        self.dense2 = nn.Linear(128, out)

        self.layers = [
            self.conv1, self.bn1, self.pool1, func.relu,
            self.conv2, self.bn2, self.pool2, func.relu,
            self.conv3, self.bn3, self.pool3, func.relu,
            nn.Flatten(),
            self.dense1, func.relu, self.dense2
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

