import itertools
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchdms.binarymap import BinarymapDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


def make_data_loader_infinite(data_loader):
    """
    With this we can always just ask for more data with next(), going through
    minibatches as guided by DataLoader.
    """
    for loader in itertools.repeat(data_loader):
        for data in loader:
            yield data


class Analysis:
    def __init__(self, model, train_data_list):
        self.learning_rate = 1e-3
        self.batch_size = 5000
        self.model = model
        self.train_datasets = [
            BinarymapDataset(train_data) for train_data in train_data_list
        ]
        self.train_loaders = [
            DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            for train_dataset in self.train_datasets
        ]
        self.train_infinite_loaders = [
            make_data_loader_infinite(train_loader)
            for train_loader in self.train_loaders
        ]
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.losses = []

    def train(self, criterion, epoch_count):
        self.losses = []
        batch_count = 1 + max(map(len, self.train_datasets)) // self.batch_size
        scheduler = ReduceLROnPlateau(self.optimizer, patience=5, verbose=True)
        self.model.train()  # Sets model to training mode.
        for _ in range(epoch_count):
            total_loss = 0
            for _ in range(batch_count):
                self.optimizer.zero_grad()
                per_loader_losses = torch.zeros(len(self.train_infinite_loaders))
                for i, train_infinite_loader in enumerate(self.train_infinite_loaders):
                    batch = next(train_infinite_loader)
                    outputs = self.model(batch["variants"])
                    per_loader_losses[i] = criterion(
                        outputs.squeeze(), batch["func_scores"]
                    ).sqrt()
                loss = torch.sum(per_loader_losses)
                self.losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            scheduler.step(total_loss)

    def evaluate(self, test_data):
        test_dataset = BinarymapDataset(test_data)
        predicted = self.model(test_dataset.variants).detach().numpy().transpose()[0]
        return pd.DataFrame(
            {"Observed": test_dataset.func_scores.numpy(), "Predicted": predicted}
        )
