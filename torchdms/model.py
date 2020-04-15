import numpy as np
import torch
import torch.nn as nn


class SingleSigmoidNet(nn.Module):
    def __init__(self, input_size, hidden1_size):
        super(SingleSigmoidNet, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden1_size, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.hidden_to_output = nn.Linear(hidden1_size, 1, bias=True)

    def forward(self, x):
        out = self.input_to_hidden(x)
        out = self.sigmoid(out)
        out = self.hidden_to_output(out)
        return out


def train_network(
    model,
    bmap_factory,
    criterion,
    optimizer,
    epoch_count,
    batch_size,
    get_train_loss=False,
):
    """ Function to train network over a number of epochs, w/ given batch size.

    Args:
        - model (Net): Network model class to be trained.
        - bmap_factory (binarymap.DataFactory): input data.
        - criterion (torch Criterion): A PyTorch criterion for training.
        - optimizer (torch Optimizer): A PyTorch optimizer for training.
        - epoch_count (int): The number of epochs (passes through training data)
        - batchsize (int): The number of samples to process in a batch.
        - get_train_loss (boolean): True if training loss by batch is desired
                                    to be returned.
    Returns:
        - model (Net): The trained Network model.
    """
    losses = []
    for epoch in range(epoch_count):
        nvariants = bmap_factory.nvariants()
        permutation = np.random.permutation(nvariants)

        for i in range(0, nvariants, batch_size):
            optimizer.zero_grad()
            idxs = permutation[i : i + batch_size]
            batch_X, batch_Y, batch_var = bmap_factory.data_of_idxs(idxs)

            # Train the model
            outputs = model(batch_X)
            loss = torch.sqrt(criterion(outputs.squeeze(), batch_Y))

            # Backprop and SGD step
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

    if get_train_loss:
        return model, losses
    else:
        return model
