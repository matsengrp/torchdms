import sys
import os
import numpy as np

from scipy import sparse
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import random
import torch
import torch.nn as nn

# Set seed for reproducibility.
random.seed(5)
# Load in data matrix and functional scores
bmap = sparse.load_npz("/Users/zorian15/Desktop/torch-dms/data/dms_simulation_150000_variants.npz")
func_scores = np.loadtxt("/Users/zorian15/Desktop/torch-dms/data/dms_simulation_150000_variants_funcscores.txt",
                         delimiter='\t')

# Convert bmap to dataframe
bmap = bmap.toarray()


def train_network(model, n_epoch, batch_size, train_data, train_labels,
                  optimizer, criterion, get_train_loss=False):
    """ Function to train network over a number of epochs, w/ given batch size.

    Args:
        - model (Net): Network model class to be trained.
        - n_epoch (int): The number of epochs (passes through training data)
        - batchsize (int): The number of samples to process in a batch.
        - train_data (ndarray): Input training data.
        - train_labels (ndarray): Corresponding labels/targets for traindata.
        - optimizer (torch Optimizer): A PyTorch optimizer for training.
        - criterion (torch Criterion): A PyTorch criterion for training.
        - get_train_loss (boolean): True if training loss by batch is desired
                                    to be returned.
    Returns:
        - model (Net): The trained Network model.
    """
    losses = []
    for epoch in range(n_epoch):
        permutation = np.random.permutation(train_data.shape[0])

        for i in range(0, train_data.shape[0], batch_size):
            optimizer.zero_grad()
            id = permutation[i:i+batch_size]
            batch_X, batch_y = train_data[id], train_labels[id]

            batch_X = torch.from_numpy(batch_X).float()
            batch_y = torch.from_numpy(batch_y).float()

            # Train the model
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)

            # Backprop and SGD step
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

    if get_train_loss:
        return model, losses
    else:
        return model


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


X_train, X_test, y_train, y_test = train_test_split(
    bmap, func_scores, test_size=0.1, random_state=5)


# Define torch versions of data
X_train_torch = torch.from_numpy(X_train).float()
y_train_torch = torch.from_numpy(y_train).float()
X_test_torch = torch.from_numpy(X_test).float()
y_test_torch = torch.from_numpy(y_test).float()


net = Net(input_size=bmap.shape[1], hidden1_size=1)
# print(net)
net.train()
criterion = torch.nn.MSELoss()  # MSE loss function
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
net, train_loss = train_network(model=net, n_epoch=100, batch_size=400,
                                train_data=X_train, train_labels=y_train,
                                optimizer=optimizer, criterion=criterion,
                                get_train_loss=True)


# Make predictions on training data and testing data.
y_pred = net(X_train_torch)
y_test_preds = net(X_test_torch)


net_loss = criterion(y_pred.squeeze(), y_train_torch)
net_test_loss = criterion(y_test_preds.squeeze(), y_test_torch)
print('Training loss:', net_loss.item())
print('Variation in training labels:', np.var(y_train))
print("Testing loss: ", net_test_loss.item())
print("Variation in testing labels:", np.var(y_test))


index = np.array(list(range(len(train_loss))))  # X-axis

plt.figure(0)
plt.plot(index, train_loss)
plt.axhline(y=np.var(y_train), color='r', linestyle='dashed',
            label="Var(y)={:.2f}".format(np.var(y_train)))
plt.xlabel("Batch #")
plt.ylabel("Loss (MSE)")
plt.title("Model Training Loss by Batch")
plt.legend(loc="upper right")
plt.savefig("/Users/zorian15/Desktop/torch-dms/figs/training_loss.png")


trained_weights = net.fc1.weight.data.numpy()

plt.figure(1)
sns.distplot(trained_weights)
plt.title("Histogram of Learned Weights")
plt.xlabel("Weight")
plt.ylabel("Frequency")
plt.savefig("/Users/zorian15/Desktop/torch-dms/figs/model_weights.png")


sns.scatterplot(x=y_train, y=y_pred.detach().numpy().squeeze())
plt.xlabel("Observed Function Score")
plt.ylabel("Predicted Function Score")
plt.title("Observed vs Predicted")
plt.savefig("/Users/zorian15/Desktop/torch-dms/figs/obs_vs_pred.png")
