"""A wrapper class for training models."""
import math
import itertools
import sys
import click
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchdms.model import monotonic_params_from_latent_space


def make_data_loader_infinite(data_loader):
    """With this we can always just ask for more data with next(), going
    through minibatches as guided by DataLoader."""
    for loader in itertools.repeat(data_loader):
        for data in loader:
            yield data


class Analysis:
    """A wrapper class for training models."""

    def __init__(
        self,
        model,
        model_path,
        val_data,
        train_data_list,
        batch_size=500,
        learning_rate=1e-3,
        device="cpu",
    ):
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.model = model
        self.model_path = model_path
        self.val_data = val_data
        self.train_datasets = train_data_list
        self.train_loaders = [
            DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            for train_dataset in train_data_list
        ]
        self.train_infinite_loaders = [
            make_data_loader_infinite(train_loader)
            for train_loader in self.train_loaders
        ]
        self.val_loss_record = sys.float_info.max

    def train(
        self, epoch_count, loss_fn, patience=10, min_lr=1e-5, loss_weight_span=None
    ):
        """Train self.model."""
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

            def loss_decays_of_target_extrema(extremum_pairs_across_targets):
                return [
                    loss_decay_of_extrema(*extremum_pair)
                    for extremum_pair in extremum_pairs_across_targets
                ]

            target_extrema_across_strata = [
                train_dataset.target_extrema() for train_dataset in self.train_datasets
            ]
            loss_decays = [
                loss_decays_of_target_extrema(extremum_pairs_across_targets)
                for extremum_pairs_across_targets in target_extrema_across_strata
            ]
            val_loss_decay = loss_decays_of_target_extrema(
                self.val_data.target_extrema()
            )
        else:
            no_loss_decay = [None] * target_count
            loss_decays = [no_loss_decay for _ in self.train_datasets]
            val_loss_decay = no_loss_decay

        batch_count = 1 + max(map(len, self.train_datasets)) // self.batch_size
        self.model.train()  # Sets model to training mode.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, patience=patience, verbose=True)
        self.model.to(self.device)

        def loss_of_targets_and_prediction(targets, predictions, per_target_loss_decay):
            """Return loss on the valid predictions, i.e. the ones that are not
            NaN."""
            valid_target_indices = torch.isfinite(targets)
            valid_targets = targets[valid_target_indices].to(self.device)
            valid_predict = predictions[valid_target_indices].to(self.device)
            return loss_fn(valid_targets, valid_predict, per_target_loss_decay)

        def complete_loss(targets, predictions, loss_decays):
            """Compute our total (across targets) loss with regularization.

            Here we compute loss separately for each target, before
            summing the results. This allows for us to take advantage of
            the samples which may contain missing information for a
            subset of the targets.
            """
            per_target_loss = [
                loss_of_targets_and_prediction(
                    targets[:, target_idx],
                    predictions[:, target_idx],
                    per_target_loss_decay,
                )
                for target_idx, per_target_loss_decay in zip(
                    range(target_count), loss_decays
                )
            ]
            return sum(per_target_loss) + self.model.regularization_loss()

        def step_model():
            per_epoch_loss = 0.0
            for _ in range(batch_count):
                optimizer.zero_grad()
                per_batch_loss = 0.0
                for train_infinite_loader, per_stratum_loss_decays in zip(
                    self.train_infinite_loaders, loss_decays
                ):

                    batch = next(train_infinite_loader)
                    samples = batch["samples"].to(self.device)
                    predictions = self.model(samples)

                    loss = complete_loss(
                        batch["targets"], predictions, per_stratum_loss_decays
                    )
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
                per_epoch_loss += per_batch_loss
                optimizer.step()

            val_samples = self.val_data.samples.to(self.device)
            val_predictions = self.model(val_samples)
            val_loss = complete_loss(
                self.val_data.targets.to(self.device), val_predictions, val_loss_decay
            ).item()
            if val_loss < self.val_loss_record:
                print(f"\nvalidation loss record: {val_loss}")
                torch.save(self.model, self.model_path)
                self.val_loss_record = val_loss

            scheduler.step(val_loss)

        with click.progressbar(range(epoch_count)) as progress_bar:
            for _ in progress_bar:
                step_model()
                if optimizer.state_dict()["param_groups"][0]["lr"] < min_lr:
                    click.echo("Learning rate dropped below stated minimum. Stopping.")
                    break

    def multi_train(
        self,
        independent_start_count,
        epoch_count,
        loss_fn,
        patience=10,
        min_lr=1e-5,
        loss_weight_span=None,
    ):
        """Do pre-training on self.model using the specified number of
        independent starts, writing the best pre-trained model to the model
        path, then fully training it."""
        pre_training_epoch_fraction = 0.1
        pre_taining_epoch_count = int(pre_training_epoch_fraction * epoch_count)
        for independent_start_idx in range(1, independent_start_count + 1):
            click.echo(
                f"LOG: Independent start {independent_start_idx}/{independent_start_count}"
            )
            self.model.randomize_parameters()
            self.train(
                pre_taining_epoch_count, loss_fn, patience, min_lr, loss_weight_span
            )
        click.echo("LOG: Beginning full training.")
        self.model = torch.load(self.model_path)
        self.train(epoch_count, loss_fn, patience, min_lr, loss_weight_span)
