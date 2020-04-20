import click
import itertools
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchdms.data import BinarymapDataset
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

    def train(self, criterion, epoch_count):
        losses = []
        batch_count = 1 + max(map(len, self.train_datasets)) // self.batch_size
        scheduler = ReduceLROnPlateau(self.optimizer, patience=5, verbose=True)
        self.model.train()  # Sets model to training mode.

        def step_model():
            per_epoch_loss = 0.0
            for _ in range(batch_count):
                self.optimizer.zero_grad()
                per_batch_loss = 0.0
                for train_infinite_loader in self.train_infinite_loaders:
                    batch = next(train_infinite_loader)
                    outputs = self.model(batch["variants"])
                    loss = criterion(outputs.squeeze(), batch["func_scores"]).sqrt()
                    per_batch_loss += loss.item()
                    # Note that here we are using gradient accumulation: calling
                    # backward for each loader before clearing the gradient via
                    # zero_grad. See, e.g. https://link.medium.com/wem03OhPH5
                    loss.backward()
                losses.append(per_batch_loss)
                per_epoch_loss += per_batch_loss
                self.optimizer.step()

            scheduler.step(per_epoch_loss)

        with click.progressbar(range(epoch_count)) as progress_bar:
            for _ in progress_bar:
                step_model()

        return losses

    def evaluate(self, test_data):
        test_dataset = BinarymapDataset(test_data)
        predicted = self.model(test_dataset.variants).detach().numpy().transpose()[0]
        return pd.DataFrame(
            {"Observed": test_dataset.func_scores.numpy(), "Predicted": predicted}
        )
