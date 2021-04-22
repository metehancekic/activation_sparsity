"""
Neural Network models for training and testing implemented in PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from ..custom_activations import TReLU_with_trainable_bias
from ..custom_layers import Normalize, take_top_k


class topk_LeNet(nn.Module):

    # 2 Conv layers, 2 Fc layers

    def __init__(self, num_classes=10, gamma=1., k=4):
        super().__init__()

        self.gamma = gamma
        self.k = k

        self.norm = Normalize(mean=[0.1307], std=[0.3081])
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,
                               stride=1, padding=2, bias=True)

        self.relu = TReLU_with_trainable_bias(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=True)
        self.fc2 = nn.Linear(1024, num_classes, bias=True)

    def bias_calculator(self, filters):
        bias = torch.sum(torch.abs(filters), dim=(1, 2, 3)).unsqueeze(dim=0)
        bias = bias.unsqueeze(dim=2)
        bias = bias.unsqueeze(dim=3)
        return bias

    def forward(self, x):

        out = self.norm(x)
        self.l1 = self.conv1(out)
        # bias = self.bias_calculator(self.conv1.weight)
        # if self.training:
        #     noise = bias * 8./255 * (torch.rand_like(self.out, device=o.device)
        #                              * self.gamma * 2 - self.gamma)
        #     self.out = self.out + noise
        out = take_top_k(self.l1, self.k)
        out = self.relu(out)

        out = F.max_pool2d(out, (2, 2))
        out = F.max_pool2d(F.relu(self.conv2(out)), (2, 2))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out
