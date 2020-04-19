import pandas as pd
import torch
from torchdms.binarymap import DataFactory
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Analysis:
    def __init__(self, model, train_data):
        self.learning_rate = 1e-3
        self.batch_size = 5000
        self.model = model
        self.train_data = train_data
        self.train_factory = DataFactory(train_data)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.losses = []

    def train(self, criterion, epoch_count):
        self.losses = []
        scheduler = ReduceLROnPlateau(self.optimizer, patience=5, verbose=True)
        self.model.train()  # Sets model to training mode.
        for _ in range(epoch_count):
            nvariants = self.train_factory.nvariants()
            permutation = torch.randperm(nvariants)
            total_loss = 0

            for i in range(0, nvariants, self.batch_size):
                self.optimizer.zero_grad()
                idxs = permutation[i : i + self.batch_size]
                batch_x, batch_y = self.train_factory.data_of_idxs(idxs)

                outputs = self.model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y).sqrt()
                self.losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            scheduler.step(total_loss)

    def evaluate(self, test_data):
        test_factory = DataFactory(test_data)
        return pd.DataFrame(
            {
                "Observed": test_factory.Y.numpy(),
                "Predicted": self.model(test_factory.X).detach().numpy().transpose()[0],
            }
        )
