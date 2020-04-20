import numpy as np
import torch
import torch.nn as nn


class SingleSigmoidNet(nn.Module):
    def __init__(self, input_size, hidden1_size=1):
        super(SingleSigmoidNet, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden1_size, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.hidden_to_output = nn.Linear(hidden1_size, 1, bias=True)

    def forward(self, x):
        out = self.input_to_hidden(x)
        out = self.sigmoid(out)
        out = self.hidden_to_output(out)
        return out


class SingleReLUNet(nn.Module):
    def __init__(self, input_size, hidden1_size=1):
        super(SingleReLUNet, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden1_size, bias=False)
        self.relu = nn.ReLU()
        self.hidden_to_output = nn.Linear(hidden1_size, 1, bias=True)

    def forward(self, x):
        out = self.input_to_hidden(x)
        out = self.relu(out)
        out = self.hidden_to_output(out)
        return out
