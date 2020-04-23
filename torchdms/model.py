import numpy as np
import torch
import torch.nn as nn


class BuildYourOwnVanillaNet(nn.Module):
    """
    Make it just how you like it.

    input size can be inferred for the train/test datasets
    and output can be inferred from the number of targets 
    specified. the rest should simply be fed in as a list
    like:
    
    layers = [2, 10, 10]

    means we have a 'latent' space of 2 nodes, connected to
    two more dense layers, each with 10 layers, before the output.
    """

    def __init__(self, input_size, layers, output_size, activation_fn=torch.sigmoid):
        super(BuildYourOwnVanillaNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        self.activation_fn = activation_fn
        for layer_index, num_nodes in enumerate(layers):
            in_size = input_size if layer_index == 0 else layers[layer_index - 1]
            bias = False if layer_index == 0 else True
            layer_name = f"custom_layer_{layer_index}"
            self.layers.append(layer_name)
            setattr(self, layer_name, nn.Linear(in_size, num_nodes, bias=bias))
        layer_name = f"custom_layer_output"
        self.layers.append(layer_name)
        setattr(self, layer_name, nn.Linear(num_nodes, output_size))

    def forward(self, x):
        out = x
        for layer_index in range(len(self.layers) - 1):
            out = self.activation_fn(getattr(self, self.layers[layer_index])(out))
        prediction = getattr(self, self.layers[-1])(out)
        return prediction


class SingleSigmoidNet(nn.Module):
    def __init__(self, input_size, hidden1_size=1):
        super(SingleSigmoidNet, self).__init__()
        self.input_size = input_size
        self.output_size = 1
        self.input_to_hidden = nn.Linear(input_size, hidden1_size, bias=False)
        self.hidden_to_output = nn.Linear(hidden1_size, 1)

    def forward(self, x):
        out = torch.sigmoid(self.input_to_hidden(x))
        out = self.hidden_to_output(out)
        return out


class AdditiveLinearModel(nn.Module):
    def __init__(self, input_size):
        super(AdditiveLinearModel, self).__init__()
        self.input_size = input_size
        self.output_size = 1
        self.input_to_output = nn.Linear(input_size, 1)

    def forward(self, x):
        out = self.input_to_output(x)
        return out


class TwoByTwoOutputTwoNet(nn.Module):
    def __init__(self, input_size):
        super(TwoByTwoOutputTwoNet, self).__init__()
        self.input_size = input_size
        self.output_size = 2
        self.input_to_hidden = nn.Linear(input_size, 2, bias=False)
        self.hidden_dense = nn.Linear(2, 2)
        self.hidden_to_output = nn.Linear(2, 2)

    def forward(self, x):
        out = torch.sigmoid(self.input_to_hidden(x))
        out = torch.sigmoid(self.hidden_dense(out))
        out = self.hidden_to_output(out)
        return out

    # TODO
    # def from_latent(self, alpha_beta):
    #    out = torch.sigmoid(self.hidden_dense(alpha_beta))
    #    out = self.hidden_to_output(out)
    #    return out


class TwoByTwoNetOutputOne(nn.Module):
    def __init__(self, input_size):
        super(TwoByTwoNet, self).__init__()
        self.input_size = input_size
        self.output_size = 1
        self.input_to_hidden = nn.Linear(input_size, 2, bias=False)
        self.hidden_dense = nn.Linear(2, 2)
        self.hidden_to_output = nn.Linear(2, 1)

    def forward(self, x):
        out = torch.sigmoid(self.input_to_hidden(x))
        out = torch.sigmoid(self.hidden_dense(out))
        out = self.hidden_to_output(out)
        return out
