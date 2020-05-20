import itertools
import click
import math
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchdms.model import monotonic_params_from_latent_space


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

    def train(
        self, epoch_count, loss_fn, patience=10, min_lr=1e-5, loss_weight_span=None
    ):

        assert len(self.train_datasets) > 0
        target_count = self.train_datasets[0].target_count()
        assert self.model.output_size == target_count

        if loss_weight_span is not None:
            assert isinstance(loss_weight_span, float)

            def loss_decay_of_extrema(worst_score, best_score):
                loss_decay = math.log(loss_weight_span) / (worst_score - best_score)
                assert loss_decay > 0.0
                if loss_decay > 1e3:
                    click.echo("WARNING: whoa, you have a big loss decay!")
                return loss_decay

            target_extrema_across_strata = [
                train_dataset.target_extrema() for train_dataset in self.train_datasets
            ]
            loss_decays = [
                [
                    loss_decay_of_extrema(*extremum_pair)
                    for extremum_pair in extremum_pairs_across_targets
                ]
                for extremum_pairs_across_targets in target_extrema_across_strata
            ]
        else:
            loss_decays = [[None] * target_count for _ in self.train_datasets]

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
                for train_infinite_loader, per_stratum_loss_decays in zip(
                    self.train_infinite_loaders, loss_decays
                ):

                    batch = next(train_infinite_loader)
                    samples = batch["samples"].to(self.device)
                    prediction = self.model(samples)

                    # Here we compute loss seperately for each target,
                    # before summing the results. This allows for us to
                    # take advantage of the samples which may contain
                    # missing information for a subset of the targets.
                    per_target_loss = []
                    for target, per_target_loss_decay in zip(
                        range(batch["targets"].shape[1]), per_stratum_loss_decays
                    ):

                        # batch["targets"] is tensor of shape (n samples, n targets) so
                        # we identify all samples for a target which are not NaN.
                        valid_target_indices = torch.isfinite(
                            batch["targets"][:, target]
                        )
                        valid_targets = batch["targets"][:, target][
                            valid_target_indices
                        ].to(self.device)
                        valid_predict = prediction[:, target][valid_target_indices].to(
                            self.device
                        )
                        this_per_target_loss = loss_fn(
                            valid_targets, valid_predict, per_target_loss_decay
                        )
                        per_target_loss.append(this_per_target_loss)

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
