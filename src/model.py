import sys
import pandas as pd
import numpy as np

from scipy import sparse
from matplotlib import pyplot as plt

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

print(sys.path)
random.seed(5)
# Load in data matrix and functional scores
bmap = sparse.load_npz("/Users/zorian15/Desktop/torch-dms/data/dms_simulation_150000_variants.npz")
func_scores = np.loadtxt("/Users/zorian15/Desktop/torch-dms/data/dms_simulation_150000_variants_funcscores.txt",
                         delimiter='\t')

# Convert bmap to dataframe
bmap = bmap.toarray()


class Net(nn.Module):

    def __init__(self, input_size, hidden1_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)  # input -> hidden layer
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden1_size, 1)  # hidden -> output layer

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid1(out)
        out = self.fc2(out)
        return out


def make_train_test_split(data, labels, train_prop=0.8):
    """ Create train-test splits from a dataset.

    Args:
        - data (ndarray): Input dataset as a ndarray
        - labels (ndarray): Corresponding labels for data.
        - test_prop (float): Decimal representing the proportion of obs to be
                             used in the test set. Default is 80%.

    Returns:
        - X_train (ndarray): The training dataset.
        - X_test (ndarray): The testing dataset.
        - y_train (ndarray): The training labels.
        - y_test (ndarray): The testing labels.
    """
    sample_space_size = data.shape[0]
    idx = np.random.randint(sample_space_size,
                            size=int(np.ceil(sample_space_size*train_prop)))
    # Create testing set
    X_train = data[idx, :]
    y_train = labels[idx]

    # Create testing set
    X_test = data[-idx, :]
    y_test = labels[-idx]

    return X_train, y_train, X_test, y_test


# Initialize model
net = Net(input_size=bmap.shape[1], hidden1_size=1)
print(net)

# Define loss criterion and optimization method
criterion = torch.nn.MSELoss()  # MSE loss function
# Try decaying learning rate
optimizer = torch.optim.SGD(net.parameters(), lr=0.00001)

# Create training and testing sets for epochs
X_train, y_train, X_test, y_test = make_train_test_split(data=bmap,
                                                         labels=func_scores)

n_epochs = 10
batch_size = 128
losses = []

for epoch in range(n_epochs):
    permutation = np.random.permutation(bmap.shape[0])

    for i in range(0, bmap.shape[0], batch_size):
        optimizer.zero_grad()
        id = permutation[i:i+batch_size]  # Double check to ensure X and y match.
        batch_X, batch_y = bmap[id], func_scores[id]

        batch_X = torch.from_numpy(batch_X).float()
        batch_y = torch.from_numpy(batch_y).float()

        # Train the model
        outputs = net(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)

        # Backprop and SGD step
        loss.backward()
        optimizer.step()

        if (i % batch_size == 0):
            losses.append(loss.item())

x = np.array(list(range(len(losses))))
losses = np.array(losses)

plt.plot(x, losses)
plt.xlabel('Batch ID')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss by Batch Size')
plt.grid(True)
plt.show()

# Validation
y_test_var = np.var(y_test)
net.eval()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()
y_pred = net(X_test)
after_train = criterion(y_pred.squeeze(), y_test)

print('Training loss:', np.mean(losses))
print('Test loss after Training:', after_train.item())
print('Variation in test labels:', y_test_var)

# Histogram of weights post-training

# Predicted values vs observed values

# Print weights and compare with
