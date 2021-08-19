"""A wrapper class for training models."""
import math
import itertools
import sys
import click
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchdms.data import BinaryMapDataset
from torchdms.utils import (
    build_beta_map,
    get_mutation_indicies,
    get_observed_training_mutations,
    make_all_possible_mutations,
)


def make_data_loader_infinite(data_loader):
    """With this we can always just ask for more data with next(), going
    through minibatches as guided by DataLoader."""
    for loader in itertools.repeat(data_loader):
        for data in loader:
            yield data


def low_rank_approximation(beta_map, beta_rank):
    """Returns low-rank approximation of beta matrix."""
    assert beta_rank > 0
    u_vecs, s_vals, v_vecs = torch.svd(torch.from_numpy(beta_map))
    # truncate S
    s_vals[beta_rank:] = 0
    # reconstruct beta-map
    beta_approx = (u_vecs.mm(torch.diag(s_vals))).mm(torch.transpose(v_vecs, 0, 1))
    return beta_approx.transpose(1, 0).flatten()


def _make_beta_matrix_low_rank(model, latent_dim, beta_rank, wtseq, alphabet):
    """Assigns low-rank beta approximations to tdms models."""
    beta_vec = model.beta_coefficients()[latent_dim].detach().clone().numpy()
    beta_map = build_beta_map(wtseq, alphabet, beta_vec)
    model.beta_coefficients()[latent_dim] = low_rank_approximation(beta_map, beta_rank)


class Analysis:
    """A wrapper class for training models."""

    def __init__(
        self,
        model,
        model_path,
        val_data,
        train_data_list,
        batch_size=500,
        learning_rate=5e-3,
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
        # Store all observed mutations
        self.training_mutations = get_observed_training_mutations(train_data_list)
        # Store WT idxs
        self.wt_idxs = val_data.wt_idxs.type(torch.LongTensor)
        # Store all observed mutations in mutant idxs
        self.mutant_idxs = get_mutation_indicies(
            self.training_mutations, self.model.alphabet
        ).type(torch.LongTensor)
        self.unseen_idxs = get_mutation_indicies(
            make_all_possible_mutations(val_data).difference(self.training_mutations),
            self.model.alphabet,
        ).type(torch.LongTensor)
        self.gauge_mask = torch.zeros(self.model.sequence_length * len(self.model.alphabet), dtype=torch.bool)
        self.gauge_mask[torch.cat((self.wt_idxs, self.unseen_idxs))] = 1
        self.model.fix_gauge(self.gauge_mask)

    def loss_of_targets_and_prediction(
        self, loss_fn, targets, predictions, per_target_loss_decay
    ):
        """Return loss on the valid predictions, i.e. the ones that are not
        NaN."""
        valid_target_indices = torch.isfinite(targets)
        valid_targets = targets[valid_target_indices].to(self.device)
        valid_predict = predictions[valid_target_indices].to(self.device)
        return loss_fn(valid_targets, valid_predict, per_target_loss_decay)

    def complete_loss(self, loss_fn, targets, predictions, loss_decays):
        """Compute our total (across targets) loss with regularization.

        Here we compute loss separately for each target, before summing
        the results. This allows for us to take advantage of the samples
        which may contain missing information for a subset of the
        targets.
        """
        per_target_loss = [
            self.loss_of_targets_and_prediction(
                loss_fn,
                targets[:, target_idx],
                predictions[:, target_idx],
                per_target_loss_decay,
            )
            for target_idx, per_target_loss_decay in zip(
                range(targets.shape[1]), loss_decays
            )
        ]
        return sum(per_target_loss) + self.model.regularization_loss()

    def train(
        self,
        epoch_count,
        loss_fn,
        patience=10,
        min_lr=1e-5,
        loss_weight_span=None,
        exp_target=None,
        beta_rank=None,
    ):
        """Train self.model using all the bells and whistles."""
        assert len(self.train_datasets) > 0
        target_count = self.train_datasets[0].target_count()
        assert self.model.output_size == target_count

        if exp_target is not None:
            loss_weight_span = None
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
        self.model.to(self.device)

        def step_model(optimizer, scheduler):
            for _ in range(batch_count):
                optimizer.zero_grad()
                per_batch_loss = 0.0
                for train_infinite_loader, per_stratum_loss_decays in zip(
                    self.train_infinite_loaders, loss_decays
                ):

                    batch = next(train_infinite_loader)
                    samples = batch["samples"].to(self.device)
                    predictions = self.model(samples)
                    loss = self.complete_loss(
                        loss_fn, batch["targets"], predictions, per_stratum_loss_decays
                    )
                    per_batch_loss += loss.item()

                    # Note that here we are using gradient accumulation: calling
                    # backward for each loader before clearing the gradient via
                    # zero_grad. See, e.g. https://link.medium.com/wem03OhPH5
                    loss.backward()

                    # if the model is monotonic, we clamp all negative parameters
                    # after the latent space ecluding all bias parameters.
                    if self.model.monotonic_sign:
                        for param in self.model.monotonic_params_from_latent_space():
                            param.data.clamp_(0)

                optimizer.step()
                self.model.fix_gauge(self.gauge_mask)
                # if k >=1, reconstruct beta matricies with truncated SVD
                if beta_rank is not None:
                    # procedure for 2D models.
                    if hasattr(self.model, "model_bind") and hasattr(
                        self.model, "model_stab"
                    ):
                        num_latent_dims = self.model.model_bind.latent_dim
                        for latent_dim in range(num_latent_dims):
                            _make_beta_matrix_low_rank(
                                self.model.model_bind,
                                latent_dim,
                                beta_rank,
                                self.val_data.wtseq,
                                self.val_data.alphabet,
                            )
                            _make_beta_matrix_low_rank(
                                self.model.model_stab,
                                latent_dim,
                                beta_rank,
                                self.val_data.wtseq,
                                self.val_data.alphabet,
                            )
                    else:
                        num_latent_dims = self.model.latent_dim
                        for latent_dim in range(num_latent_dims):
                            _make_beta_matrix_low_rank(
                                self.model,
                                latent_dim,
                                beta_rank,
                                self.val_data.wtseq,
                                self.val_data.alphabet,
                            )

            val_samples = self.val_data.samples.to(self.device)
            val_predictions = self.model(val_samples)
            val_loss = self.complete_loss(
                loss_fn,
                self.val_data.targets.to(self.device),
                val_predictions,
                val_loss_decay,
            ).item()
            if val_loss < self.val_loss_record:
                print(f"\nvalidation loss record: {val_loss}")
                torch.save(self.model, self.model_path)
                self.val_loss_record = val_loss

            scheduler.step(val_loss)

        for training_style in self.model.training_style_sequence:
            training_style()
            self.val_loss_record = sys.float_info.max
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            scheduler = ReduceLROnPlateau(optimizer, patience=patience, verbose=True)

            with click.progressbar(range(epoch_count)) as progress_bar:
                for _ in progress_bar:
                    step_model(optimizer, scheduler)
                    if optimizer.state_dict()["param_groups"][0]["lr"] < min_lr:
                        click.echo(
                            "Learning rate dropped below stated minimum. Stopping."
                        )
                        break

    def multi_train(
        self,
        independent_start_count,
        independent_start_epoch_count,
        epoch_count,
        loss_fn,
        patience=10,
        min_lr=1e-5,
        loss_weight_span=None,
        exp_target=None,
        beta_rank=None,
    ):
        """Do pre-training on self.model using the specified number of
        independent starts, writing the best pre-trained model to the model
        path, then fully training it."""
        if independent_start_epoch_count is None:
            independent_start_epoch_count = math.ceil(0.1 * epoch_count)
        for independent_start_idx in range(1, independent_start_count + 1):
            click.echo(
                f"LOG: Independent start {independent_start_idx}/{independent_start_count}"
            )
            self.model.randomize_parameters()
            self.train(
                independent_start_epoch_count,
                loss_fn,
                patience,
                min_lr,
                loss_weight_span,
            )
        click.echo("LOG: Beginning full training.")
        self.model = torch.load(self.model_path)
        self.train(
            epoch_count,
            loss_fn,
            patience,
            min_lr,
            loss_weight_span,
            exp_target,
            beta_rank,
        )

    def simple_train(self, epoch_count, loss_fn):
        """Bare-bones training of self.model.

        This traning doesn't even handle nans. If you want that behavior, just use
        self.loss_of_targets_and_prediction rather than loss_fn directly.

        We also cat together all of the data rather than getting gradients on a
        per-stratum basis. If you don't want this behavior use
        self.train_infinite_loaders rather than the train_infinite_loaders defined
        below.
        """
        assert len(self.train_datasets) > 0
        target_count = self.train_datasets[0].target_count()
        assert self.model.output_size == target_count

        batch_count = 1 + max(map(len, self.train_datasets)) // self.batch_size
        self.model.train()  # Sets model to training mode.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.to(self.device)

        train_infinite_loaders = [
            make_data_loader_infinite(
                DataLoader(
                    BinaryMapDataset.cat(self.train_datasets),
                    batch_size=self.batch_size,
                    shuffle=True,
                )
            )
        ]

        def step_model():
            for _ in range(batch_count):
                optimizer.zero_grad()
                for train_infinite_loader in train_infinite_loaders:
                    batch = next(train_infinite_loader)
                    samples = batch["samples"].to(self.device)
                    predictions = self.model(samples)
                    loss = loss_fn(batch["targets"], predictions)

                    # Note that here we are using gradient accumulation: calling
                    # backward for each loader before clearing the gradient via
                    # zero_grad. See, e.g. https://link.medium.com/wem03OhPH5
                    loss.backward()

                    # if the model is monotonic, we clamp all negative parameters
                    # after the latent space excluding all bias parameters.
                    if self.model.monotonic_sign:
                        for param in self.model.monotonic_params_from_latent_space():
                            param.data.clamp_(0)
                optimizer.step()

        with click.progressbar(range(epoch_count)) as progress_bar:
            for _ in progress_bar:
                step_model()

        torch.save(self.model, self.model_path)
