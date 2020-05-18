import itertools
import click
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchdms.utils import monotonic_params_from_latent_space


def make_data_loader_infinite(data_loader):
    """
    With this we can always just ask for more data with next(), going through
    minibatches as guided by DataLoader.
    """
    for loader in itertools.repeat(data_loader):
        for data in loader:
            yield data


class Analysis:
    def __init__(
        self, model, train_data_list, batch_size=500, learning_rate=1e-3, device="cpu"
    ):
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.model = model
        self.train_datasets = train_data_list
        self.train_loaders = [
            DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            for train_dataset in train_data_list
        ]
        self.train_infinite_loaders = [
            make_data_loader_infinite(train_loader)
            for train_loader in self.train_loaders
        ]
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

    def train(self, epoch_count, loss_fn, patience=10, min_lr=1e-6):

        # TODO assert self.model.output_size == len(self.train_datasets[0].targets)

        losses_history = []
        batch_count = 1 + max(map(len, self.train_datasets)) // self.batch_size
        scheduler = ReduceLROnPlateau(self.optimizer, patience=patience, verbose=True)
        self.model.train()  # Sets model to training mode.
        self.model.to(self.device)

        def step_model():
            per_epoch_loss = 0.0
            for _ in range(batch_count):
                self.optimizer.zero_grad()
                per_batch_loss = 0.0
                for train_infinite_loader in self.train_infinite_loaders:

                    batch = next(train_infinite_loader)
                    samples = batch["samples"].to(self.device)
                    prediction = self.model(samples)

                    # Here we compute loss seperately for each target,
                    # before summing the results. This allows for us to
                    # take advantage of the samples which may contain
                    # missing information for a subset of the targets.
                    per_target_loss = []
                    for target in range(batch["targets"].shape[1]):

                        # batch["targets"] is tensor of shape
                        # (n samples, n targets) so we identify
                        # all samples for a target which are not NaN.
                        valid_targets_index = torch.isfinite(
                            batch["targets"][:, target]
                        )
                        valid_targets = batch["targets"][:, target][
                            valid_targets_index
                        ].to(self.device)
                        valid_predict = prediction[:, target][valid_targets_index].to(
                            self.device
                        )
                        per_target_loss.append(loss_fn(valid_targets, valid_predict))

                    loss = sum(per_target_loss) + self.model.regularization_loss()
                    per_batch_loss += loss.item()

                    # Note that here we are using gradient accumulation: calling
                    # backward for each loader before clearing the gradient via
                    # zero_grad. See, e.g. https://link.medium.com/wem03OhPH5
                    loss.backward()

                    # if the model is monotonic, we clamp all negative parameters
                    # after the latent space ecluding all bias parameters.
                    if self.model.monotonic_sign:
                        for param in monotonic_params_from_latent_space(self.model):
                            param.data.clamp_(0)
                losses_history.append(per_batch_loss)
                per_epoch_loss += per_batch_loss
                self.optimizer.step()

            scheduler.step(per_epoch_loss)

        with click.progressbar(range(epoch_count)) as progress_bar:
            for _ in progress_bar:
                step_model()
                if self.optimizer.state_dict()["param_groups"][0]["lr"] < min_lr:
                    click.echo("Learning rate dropped below stated minimum. Stopping.")
                    break

        return losses_history
