"""Small causal temporal conv net. The ridge decoder is the one to train first."""

import torch
from torch import nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, signal):
        return self.conv(nn.functional.pad(signal, (self.pad, 0)))


class AngleTCN(nn.Module):
    def __init__(self, n_joints=14, channels=8, hidden=32, kernel_size=5, dropout=0.1):
        super().__init__()
        self.block1 = nn.Sequential(CausalConv1d(channels, hidden, kernel_size), nn.ReLU(),
                                    nn.Dropout(dropout))
        self.block2 = nn.Sequential(CausalConv1d(hidden, hidden, kernel_size), nn.ReLU(),
                                    nn.Dropout(dropout))
        self.head = nn.Linear(hidden, n_joints)

    def forward_sequence(self, signal):
        hidden = self.block2(self.block1(signal))
        return self.head(hidden.transpose(1, 2))

    def forward(self, signal):
        return self.forward_sequence(signal)[:, -1]
