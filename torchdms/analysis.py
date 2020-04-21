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
    def __init__(self, model, train_data_list, batch_size=500, learning_rate=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
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

    def train(self, criterion, epoch_count, min_lr=1e-6):
        losses = []
        batch_count = 1 + max(map(len, self.train_datasets)) // self.batch_size
        scheduler = ReduceLROnPlateau(self.optimizer, patience=5, verbose=True)
        self.model.train()  # Sets model to training mode.
        self.model.to(self.device)

        def step_model():
            per_epoch_loss = 0.0
            for _ in range(batch_count):
                self.optimizer.zero_grad()
                per_batch_loss = 0.0
                for train_infinite_loader in self.train_infinite_loaders:
                    batch = next(train_infinite_loader)
                    variants = batch["variants"].to(self.device)
                    func_scores = batch["func_scores"].to(self.device)
                    outputs = self.model(variants)
                    loss = criterion(outputs.squeeze(), func_scores).sqrt()
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
                if self.optimizer.state_dict()["param_groups"][0]["lr"] < min_lr:
                    click.echo("Learning rate dropped below stated minimum. Stopping.")
                    break

        return losses

    def evaluate(self, test_data):
        self.model.eval()
        self.model.to(self.device)
        test_dataset = BinarymapDataset(test_data)
        variants = test_dataset.variants.to(self.device)
        predicted = self.model(variants).detach().numpy().transpose()[0]
        return pd.DataFrame(
            {
                "Observed": test_dataset.func_scores.numpy(),
                "Predicted": predicted,
                "n_aa_substitutions": test_data.n_aa_substitutions,
            }
        )

    def process_evaluation(self, results):
        corr = results.corr().iloc[0, 1]
        ax = results.plot.scatter(
            x="Observed", y="Predicted", c=results["n_aa_substitutions"], cmap="viridis"
        )
        ax.text(0, 0.95 * max(results["Predicted"]), f"corr = {corr:.3f}")
        return corr, ax
