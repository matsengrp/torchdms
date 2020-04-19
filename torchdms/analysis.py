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
    def __init__(self, model, train_data):
        self.learning_rate = 1e-3
        self.batch_size = 5000
        self.model = model
        self.train_dataset = BinarymapDataset(train_data)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.train_infinite_dataloader = make_data_loader_infinite(
            self.train_dataloader
        )
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.losses = []

    def train(self, criterion, epoch_count):
        self.losses = []
        scheduler = ReduceLROnPlateau(self.optimizer, patience=5, verbose=True)
        self.model.train()  # Sets model to training mode.
        for _ in range(epoch_count):
            total_loss = 0
            batch_count = 1 + len(self.train_dataset) // self.batch_size
            for _ in range(batch_count):
                self.optimizer.zero_grad()
                batch = next(self.train_infinite_dataloader)
                outputs = self.model(batch["variants"])
                loss = criterion(outputs.squeeze(), batch["func_scores"]).sqrt()
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
