#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """LeNet architecture for MNIST dataset"""
    def __init__(self):
        super(LeNet, self).__init__()
        # First convolutional layer: 1 input channel, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # Second convolutional layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # First conv layer followed by ReLU and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # Second conv layer followed by ReLU and max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # Flatten the tensor
        x = x.view(-1, 16 * 4 * 4)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_params(self):
        """Return model parameters as a list of tensors"""
        return [p.data.clone() for p in self.parameters()]

    def set_params(self, params):
        """Set model parameters from a list of tensors"""
        for p, new_p in zip(self.parameters(), params):
            p.data = new_p.clone().to(p.device) 