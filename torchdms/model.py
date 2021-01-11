"""Our models."""
from abc import abstractmethod
import re
import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchdms.utils import from_pickle_file


def identity(x):
    """The identity function, to be used as "no activation."."""
    return x


class TorchdmsModel(nn.Module):
    """A superclass for our models to combine shared behavior."""

    def __init__(self, input_size, target_names, alphabet, freeze_betas=False):
        super().__init__()
        self.input_size = input_size
        self.target_names = target_names
        self.output_size = len(target_names)
        self.alphabet = alphabet
        self.freeze_betas = freeze_betas
        self.monotonic_sign = None
        self.layers = []

    def __str__(self):
        return super().__str__() + "\n" + self.characteristics.__str__()

    @property
    @abstractmethod
    def characteristics(self):
        pass

    @abstractmethod
    def forward(self, x):  # pylint: disable=arguments-differ
        pass

    @abstractmethod
    def regularization_loss(self):
        pass

    @abstractmethod
    def str_summary(self):
        pass

    @abstractmethod
    def to_latent(self, x):
        pass

    @property
    def sequence_length(self):
        alphabet_length = len(self.alphabet)
        assert self.input_size % alphabet_length == 0
        return self.input_size // alphabet_length

    def numpy_single_mutant_predictions(self):
        """Return the single mutant predictions as a numpy array of shape (AAs,
        sites, outputs)."""
        input_tensor = torch.zeros((1, self.input_size))

        def forward_on_ith_basis_vector(i):
            input_tensor[0, i] = 1.0
            return_value = self.forward(input_tensor).detach().numpy()
            input_tensor[0, i] = 0.0
            return return_value

        # flat_results first indexes through the outputs, then the alphabet, then the
        # sites.
        flat_results = np.concatenate(
            [forward_on_ith_basis_vector(i) for i in range(input_tensor.shape[1])]
        )

        # Reshape takes its arguments from right to left. So in this case:
        # - take entries until we have output_size of them
        # - take those until we have len(alphabet) of them
        # ... and we will have sequence_length of those.
        # This is the transpose of what we want, so we transpose the first two items.
        return flat_results.reshape(
            (self.sequence_length, len(self.alphabet), self.output_size)
        ).transpose(1, 0, 2)

    def single_mutant_predictions(self):
        """Return the single mutant predictions as a list (across outputs) of
        Pandas dataframes."""
        numpy_predictions = self.numpy_single_mutant_predictions()
        return [
            pd.DataFrame(
                numpy_predictions[:, :, output_idx],
                index=pd.Index(self.alphabet, name="AA"),
                columns=pd.Index(range(self.sequence_length), name="site"),
            )
            for output_idx in range(numpy_predictions.shape[-1])
        ]

    def monotonic_params_from_latent_space(self):
        """Returns all the parameters to be floored to zero in a monotonic
        model. This is every parameter after the latent space excluding bias
        parameters.

        We follow the hueristic that the latent layers of a network are
        named 'latent_layer*' and the weight bias are denoted:

        layer_name*.weight
        layer_name*.bias.

        Layers from nested modules will be prefixed (e.g. with Dianthum).
        """
        for name, param in self.named_parameters():
            parse_name = name.split(".")
            # NOTE: minus indices allow for prefixes from nested module params
            is_latent_layer = parse_name[-2] == "latent_layer"
            is_bias = parse_name[-1] == "bias"
            if not is_latent_layer and not is_bias:
                yield param

    def reflect_monotonic_params(self):
        """Ensure that monotonic parameters are non-negative by flipping their
        sign."""
        for param in self.monotonic_params_from_latent_space():
            # https://pytorch.org/docs/stable/nn.html#linear
            # because the original distribution is
            # uniform between (-k, k) where k = 1/input_features,
            # we can simply transform all weights < 0 to their positive
            # counterpart
            param.data[param.data < 0] *= -1

    def randomize_parameters(self):
        """Randomize model parameters."""
        for layer_name in self.layers:
            if layer_name != "latent_layer" or not self.freeze_betas:
                getattr(self, layer_name).reset_parameters()

        if self.monotonic_sign is not None:
            self.reflect_monotonic_params()

    def default_training_style(self):
        """The default training style."""
        click.echo("Training in default style.")
        return self.parameters()

    @property
    def training_styles(self):
        return [self.default_training_style]


class LinearModel(TorchdmsModel):
    """The simplest model."""

    def __init__(
        self,
        input_size,
        target_names,
        alphabet,
    ):
        super().__init__(input_size, target_names, alphabet)
        self.latent_layer = nn.Linear(self.input_size, self.output_size)
        self.layers = ["latent_layer"]

    @property
    def characteristics(self):
        return {}

    def forward(self, x):  # pylint: disable=arguments-differ
        return self.layer(x)

    def regularization_loss(self):
        return 0.0

    def str_summary(self):
        return "Linear"

    def to_latent(self, x):
        return self.forward(x)


class Planifolia(TorchdmsModel):
    """Make it just how you like it.

    input size can be inferred for the train/test datasets
    and output can be inferred from the number of targets
    specified. the rest should simply be fed in as a list
    like:

    layer_sizes = [10, 2, 10, 10]
    activations = [relu, identity, relu, relu]

    means we have a 'latent' space of 2 nodes, feeding into
    two more dense layers, each with 10 nodes, before the output.
    The first layer that corresponds to an `identity` activation is the latent
    space.
    Layers before the latent layer are a nonlinear module for site-wise
    interactions, in this case one layer of 10 nodes.

    `activations` is a list of torch activations, which can be identity for no
    activation. Activations just happen between the hidden layers, and not at the output
    layer (so we aren't restricted by the range of the activation).

    If layers is fed an empty list, the model will be a
    neural network implementation of the additive linear model.
    """

    def __init__(
        self,
        input_size,
        layer_sizes,
        activations,
        target_names,
        alphabet,
        monotonic_sign=None,
        beta_l1_coefficient=0.0,
        interaction_l1_coefficient=0.0,
        freeze_betas=False,
    ):
        super().__init__(input_size, target_names, alphabet)
        self.monotonic_sign = monotonic_sign
        self.layers = []
        self.activations = activations
        self.beta_l1_coefficient = beta_l1_coefficient
        self.interaction_l1_coefficient = interaction_l1_coefficient
        self.freeze_betas = freeze_betas

        if not len(layer_sizes) == len(activations):
            raise ValueError(
                f"{len(layer_sizes)} layer sizes inconsistent with {len(activations)} activations"
            )

        try:
            self.latent_idx = self.activations.index(identity)
        except ValueError:
            self.latent_idx = 0

        # additive model
        if len(layer_sizes) == 0:
            layer_name = "latent_layer"
            self.layers.append(layer_name)
            setattr(self, layer_name, nn.Linear(input_size, self.output_size))

        # all other models
        else:
            prefix = "interaction"
            # the pre-latent interaction network should have zero bias
            bias = False
            for layer_index, num_nodes in enumerate(layer_sizes):
                if layer_index == self.latent_idx:
                    layer_name = "latent_layer"
                    # The latent layer has a bias representing wild type latent phenotype
                    # And post-latent layer has bias, since data may not be normalized to wild type
                    bias = True
                    if layer_index > 0:
                        # skip connection
                        input_size += self.input_size
                    prefix = "nonlinearity"
                else:
                    layer_name = f"{prefix}_{layer_index}"

                self.layers.append(layer_name)
                setattr(self, layer_name, nn.Linear(input_size, num_nodes, bias=bias))
                input_size = layer_sizes[layer_index]

            # final layer
            layer_name = "output_layer"
            self.layers.append(layer_name)
            setattr(self, layer_name, nn.Linear(layer_sizes[-1], self.output_size))

        if self.monotonic_sign is not None:
            # If monotonic, we want to initialize all parameters
            # which will be floored at 0, to a value above zero.
            self.reflect_monotonic_params()

    @property
    def characteristics(self):
        """Return salient characteristics of the model that aren't represented
        in the PyTorch description."""
        return {
            "activations": str(
                [activation.__name__ for activation in self.activations]
            ),
            "monotonic": self.monotonic_sign,
            "beta_l1_coefficient": self.beta_l1_coefficient,
            "interaction_l1_coefficient": self.interaction_l1_coefficient,
        }

    @property
    def internal_layer_dimensions(self):
        return [getattr(self, layer).out_features for layer in self.layers[:-1]]

    @property
    def is_linear(self):
        return len(self.internal_layer_dimensions) == 0

    @property
    def latent_dim(self):
        dims = self.internal_layer_dimensions
        if len(dims) == 0:
            return 0
        # else:
        return dims[self.latent_idx]

    def str_summary(self):
        """A one-line summary of the model."""
        if self.is_linear:
            return "linear"
        # else:
        activation_names = [activation.__name__ for activation in self.activations]
        if self.monotonic_sign is None:
            monotonic = "non-mono"
        else:
            monotonic = "mono"
        return f"{monotonic};" + ";".join(
            [
                f"{dim};{name}"
                for dim, name in zip(self.internal_layer_dimensions, activation_names)
            ]
        )

    def to_latent(self, x):
        """Map input into latent space."""
        out = x
        if self.latent_idx > 0:
            # loop over pre-latent interaction layers
            for layer_name, activation in zip(
                self.layers[: self.latent_idx], self.activations[: self.latent_idx]
            ):
                out = activation(getattr(self, layer_name)(out))
            # skip connection concatenates single-mutant effects and interaction
            # features for the latent layer
            out = torch.cat((x, out), 1)

        return self.activations[self.latent_idx](
            getattr(self, self.layers[self.latent_idx])(out)
        )

    def from_latent_to_output(self, x):
        """Evaluate the mapping from the latent space to output."""
        if self.is_linear:
            return x
        # else:
        out = x
        for layer_name, activation in zip(
            self.layers[self.latent_idx + 1 : -1],
            self.activations[self.latent_idx + 1 :],
        ):
            out = activation(getattr(self, layer_name)(out))
        # The last layer acts without an activation, which is on purpose because we
        # don't want to be limited to the range of the activation.
        out = getattr(self, self.layers[-1])(out)
        if self.monotonic_sign:
            out *= self.monotonic_sign
        return out

    def forward(self, x):  # pylint: disable=arguments-differ
        """Compose data --> latent --> output."""
        return self.from_latent_to_output(self.to_latent(x))

    def beta_coefficients(self):
        """Beta coefficients (single mutant effects only, no interaction
        terms)"""
        # This implementation assumes the single mutant terms are indexed first
        return self.latent_layer.weight.data[:, : self.input_size]

    def regularization_loss(self):
        """L1 penalize single mutant effects, and pre-latent interaction
        weights."""
        penalty = 0.0
        if self.beta_l1_coefficient > 0.0:
            penalty += self.beta_l1_coefficient * self.latent_layer.weight[
                :, : self.input_size
            ].norm(1)
        if self.interaction_l1_coefficient > 0.0:
            for interaction_layer in self.layers[: self.latent_idx]:
                penalty += self.interaction_l1_coefficient * getattr(
                    self, interaction_layer
                ).weight.norm(1)
        return penalty


class Dianthum(TorchdmsModel):
    """Parallel and independent Planifolia for each of two output dimensions.

    beta_l1_coefficients and interaction_l1_coefficients are each lists
    with two elements, a penalty parameter for each of the parallel
    models
    """

    def __init__(
        self,
        input_size,
        layer_sizes,
        activations,
        target_names,
        alphabet,
        monotonic_sign=None,
        beta_l1_coefficients=None,
        interaction_l1_coefficients=None,
    ):
        super().__init__(input_size, target_names, alphabet)

        if not len(layer_sizes) == len(activations):
            raise ValueError(
                f"{len(layer_sizes)} layer sizes inconsistent with {len(activations)} activations"
            )
        if not len(layer_sizes) > 0:
            raise ValueError("must specify at least one layer")
        if not self.output_size == 2:
            raise ValueError(
                f"2D target data required for this model, got {self.output_size}D"
            )

        if beta_l1_coefficients is None:
            beta_l1_coefficients = [0.0, 0.0]
        if interaction_l1_coefficients is None:
            interaction_l1_coefficients = [0.0, 0.0]

        for i, model in enumerate(("bind", "stab")):
            self.add_module(
                f"model_{model}",
                Planifolia(
                    input_size,
                    layer_sizes,
                    activations,
                    [target_names[i]],
                    alphabet,
                    monotonic_sign,
                    beta_l1_coefficients[i],
                    interaction_l1_coefficients[i],
                ),
            )
            for layer_name in getattr(self, f"model_{model}").layers:
                layer_name_expanded = f"{layer_name}_{model}"
                setattr(
                    self,
                    layer_name_expanded,
                    getattr(getattr(self, f"model_{model}"), layer_name),
                )
                self.layers.append(layer_name_expanded)

    @property
    def characteristics(self):
        return self.model_bind.characteristics

    @property
    def internal_layer_dimensions(self):
        return self.model_bind.internal_layer_dimensions

    @property
    def is_linear(self):
        return self.internal_layer_dimensions.is_linear

    @property
    def latent_dim(self):
        return self.model_bind.latent_dim + self.model_stab.latent_dim

    def str_summary(self):
        return (
            f"{self.__class__.__name__}: ({self.model_bind.str_summary()}, "
            f"{self.model_stab.str_summary()})"
        )

    def from_latent_to_output(self, x):
        return torch.cat(
            (
                self.model_bind.from_latent_to_output(x[:, 0, None]),
                self.model_stab.from_latent_to_output(x[:, 1, None]),
            ),
            1,
        )

    def to_latent(self, x):
        return torch.cat(
            (self.model_bind.to_latent(x), self.model_stab.to_latent(x)), 1
        )

    def forward(self, x):  # pylint: disable=arguments-differ
        return self.from_latent_to_output(self.to_latent(x))

    def beta_coefficients(self):
        """beta coefficients, skip coefficients only if interaction module."""
        beta_coefficients_data = torch.cat(
            (
                self.model_bind.latent_layer.weight.data,
                self.model_stab.latent_layer.weight.data,
            )
        )
        return beta_coefficients_data[:, : self.input_size]

    def regularization_loss(self):
        """L1-penalize weights."""
        return (
            self.model_bind.regularization_loss()
            + self.model_stab.regularization_loss()
        )


class Argus(Dianthum):
    """Allows the latent space for the second output feature (i.e. stability)
    to feed forward into the network for the first output feature (i.e.
    binding).

    Diagram:
    https://user-images.githubusercontent.com/1173298/89943302-d4524d00-dbd2-11ea-827d-6ad6c238ff52.png
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # expand input dimension of first post-latent layer in bind network
        # to accommodate stab-->bind interaction
        first_post_latent_layer_idx = self.model_bind.latent_idx + 1
        layer_name = self.model_bind.layers[first_post_latent_layer_idx]
        setattr(
            self.model_bind,
            layer_name,
            nn.Linear(
                self.latent_dim,
                self.model_bind.internal_layer_dimensions[first_post_latent_layer_idx],
                bias=True,
            ),
        )

    def from_latent_to_output(self, x):
        return torch.cat(
            (
                self.model_bind.from_latent_to_output(x),
                self.model_stab.from_latent_to_output(x[:, 1, None]),
            ),
            1,
        )

    def only_train_bind_style(self):
        click.echo("Only training bind.")
        return self.model_bind.parameters()

    def only_train_stab_style(self):
        click.echo("Only training stab.")
        return self.model_stab.parameters()


class ArgusSequential(Argus):
    """Argus with sequential training: stab then bind."""

    @property
    def training_styles(self):
        return [self.only_train_stab_style, self.only_train_bind_style]


KNOWN_MODELS = {
    "Linear": LinearModel,
    "Planifolia": Planifolia,
    "Dianthum": Dianthum,
    "Argus": Argus,
    "ArgusSequential": ArgusSequential,
}


def activation_of_string(string):
    if string == "identity":
        return identity
    # else:
    if hasattr(torch, string):
        return getattr(torch, string)
    # else:
    if hasattr(torch.nn.functional, string):
        return getattr(torch.nn.functional, string)
    # else:
    raise IOError(f"Don't know activation named {string}.")


def model_of_string(model_string, data_path, **kwargs):
    """Build a model out of a string specification."""
    try:
        model_regex = re.compile(r"(.*)\((.*)\)")
        match = model_regex.match(model_string)
        model_name = match.group(1)
        arguments = match.group(2).split(",")
        if arguments == [""]:
            arguments = []
        if len(arguments) % 2 != 0:
            raise IOError
        layers = list(map(int, arguments[0::2]))
        activations = list(map(activation_of_string, arguments[1::2]))
    except Exception:
        click.echo(
            f"ERROR: Couldn't parse model description: '{model_string}'."
            "The number of arguments to a model specification must be "
            "even, alternating between layer sizes and activations."
        )
        raise
    if model_name not in KNOWN_MODELS:
        raise IOError(model_name + " not known")
    data = from_pickle_file(data_path)
    test_dataset = data.test
    if model_name == "Planifolia":
        if len(layers) == 0:
            click.echo("LOG: No layers provided, so I'm creating a linear model.")
        model = Planifolia(
            test_dataset.feature_count(),
            layers,
            activations,
            test_dataset.target_names,
            alphabet=test_dataset.alphabet,
            **kwargs,
        )
    elif model_name == "Linear":
        model = LinearModel(
            test_dataset.feature_count(),
            test_dataset.target_names,
            alphabet=test_dataset.alphabet,
        )
    elif model_name in ("Dianthum", "Argus"):
        model = KNOWN_MODELS[model_name](
            test_dataset.feature_count(),
            layers,
            activations,
            test_dataset.target_names,
            alphabet=test_dataset.alphabet,
            **kwargs,
        )
    else:
        raise NotImplementedError(model_name)
    return model
