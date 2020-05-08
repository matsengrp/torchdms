import numpy as np
import torch
import torch.nn as nn


class DMSFeedForwardModel(nn.Module):
    """
    Make it just how you like it.

    input size can be inferred for the train/test datasets
    and output can be inferred from the number of targets
    specified. the rest should simply be fed in as a list
    like:

    layers = [2, 10, 10]

    means we have a 'latent' space of 2 nodes, connected to
    two more dense layers, each with 10 layers, before the output.

    If layers is fed an empty list, the model will be a
    neural network equivilent of the Additive linear model.
    """

    def __init__(
        self,
        input_size,
        layers,
        output_size,
        activation_fn=torch.sigmoid,
        monotonic=False,
    ):
        super(DMSFeedForwardModel, self).__init__()
        self.monotonic = monotonic
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        self.activation_fn = activation_fn

        layer_name = f"input_layer"

        # additive model
        if len(layers) == 0:
            self.layers.append(layer_name)
            setattr(self, layer_name, nn.Linear(input_size, output_size))

        # all other models
        else:
            # all internal layers
            in_size = input_size
            bias = False
            for layer_index, num_nodes in enumerate(layers):
                self.layers.append(layer_name)
                setattr(self, layer_name, nn.Linear(in_size, num_nodes, bias=bias))
                layer_name = f"internal_layer_{layer_index}"
                in_size = layers[layer_index]
                bias = True

            # final layer
            layer_name = f"output_layer"
            self.layers.append(layer_name)
            setattr(self, layer_name, nn.Linear(num_nodes, output_size))

    def forward(self, x):
        out = x
        for layer_index in range(len(self.layers) - 1):
            out = self.activation_fn(getattr(self, self.layers[layer_index])(out))
        prediction = getattr(self, self.layers[-1])(out)
        return prediction

    def from_latent(self, x):
        assert len(self.layers) != 0
        out = x
        for layer_index in range(1, len(self.layers) - 1):
            out = self.activation_fn(getattr(self, self.layers[layer_index])(out))
        prediction = getattr(self, self.layers[-1])(out)
        return prediction


class SingleSigmoidNet(nn.Module):
    def __init__(self, input_size, hidden1_size=1, monotonic=False):
        super(SingleSigmoidNet, self).__init__()
        self.monotonic = monotonic
        self.input_size = input_size
        self.output_size = 1
        self.input_to_hidden = nn.Linear(input_size, hidden1_size, bias=False)
        self.hidden_to_output = nn.Linear(hidden1_size, 1)

    def forward(self, x):
        out = torch.sigmoid(self.input_to_hidden(x))
        out = self.hidden_to_output(out)
        return out


class AdditiveLinearModel(nn.Module):
    def __init__(self, input_size, monotonic=False):
        super(AdditiveLinearModel, self).__init__()
        self.monotonic = monotonic
        self.input_size = input_size
        self.output_size = 1
        self.input_to_output = nn.Linear(input_size, 1)

    def forward(self, x):
        out = self.input_to_output(x)
        return out


class TwoByTwoOutputTwoNet(nn.Module):
    def __init__(self, input_size, monotonic=False):
        super(TwoByTwoOutputTwoNet, self).__init__()
        self.monotonic = monotonic
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
    def __init__(self, input_size, monotonic=False):
        super(TwoByTwoNet, self).__init__()
        self.monotonic = monotonic
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
