import numpy as np
import torch
import torch.nn as nn


class SingleSigmoidNet(nn.Module):
    def __init__(self, input_size, hidden1_size=1):
        super(SingleSigmoidNet, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden1_size, bias=False)
        self.hidden_to_output = nn.Linear(hidden1_size, 1)

    def forward(self, x):
        out = torch.sigmoid(self.input_to_hidden(x))
        out = self.hidden_to_output(out)
        return out


class TwoByOneNet(nn.Module):
    def __init__(self, input_size):
        super(TwoByOneNet, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, 2, bias=False)
        self.hidden_to_output = nn.Linear(2, 1)

    def forward(self, x):
        out = torch.sigmoid(self.input_to_hidden(x))
        out = self.hidden_to_output(out)
        return out


class TwoByTwoNet(nn.Module):
    def __init__(self, input_size):
        super(TwoByTwoNet, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, 2, bias=False)
        self.hidden_dense = nn.Linear(2, 2)
        self.hidden_to_output = nn.Linear(2, 1)

    def forward(self, x):
        out = torch.sigmoid(self.input_to_hidden(x))
        out = torch.sigmoid(self.hidden_dense(out))
        out = self.hidden_to_output(out)
        return out
