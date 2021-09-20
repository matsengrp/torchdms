r"""Generalized global epistasis models."""
from typing import List, Tuple, Callable, Dict, Generator, Optional
from abc import ABC, abstractmethod
import re
import click
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchdms.utils import from_pickle_file


class TorchdmsModel(ABC, nn.Module):
    r"""An abstract superclass for our models to combine shared behavior.

    Args:
        input_size: number of one-hot sequence indicators :math:`L\times |\mathcal{A}|`,
                    for sequence of length :math:`L` and amino acid alphabet :math:`\mathcal{A}`.
        target_names: output names (e.g. ``"expression"``, ``"stability"``).
        alphabet: amino acid alphabet :math:`\mathcal{A}`.
        monotonic_sign: ``1`` or ``-1`` for monotonicity, or ``None``
    """

    def __init__(
        self,
        input_size: int,
        target_names: List[str],
        alphabet: str,
        monotonic_sign: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.target_names = target_names
        self.alphabet = alphabet
        self.monotonic_sign = monotonic_sign

        self.output_size: int = len(target_names)
        """output dimension (number of phenotypes)"""
        self.layers: List[int] = []
        """list of layer names"""
        self.training_style_sequence: List[Callable] = [self._default_training_style]
        """list of training style functions that (de)activate gradients in submodules"""

    def __str__(self):
        return super().__str__() + "\n" + self.characteristics.__str__()

    @property
    @abstractmethod
    def characteristics(self) -> Dict:
        r"""Salient characteristics of the model that aren't represented
        in the PyTorch description."""

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Number of dimensions in latent space."""

    @abstractmethod
    def regularization_loss(self) -> torch.Tensor:
        r"""Compute regularization penalty, a scalar-valued :py:class:`torch.Tensor`"""

    @abstractmethod
    def str_summary(self) -> str:
        r"""A one-line summary of the model."""

    @abstractmethod
    def beta_coefficients(self) -> torch.Tensor:
        r"""Single mutant effects :math:`\beta`"""

    @abstractmethod
    def to_latent(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Latent space representation :math:`Z`

        .. math::
            z \equiv \phi(x) \equiv \beta^\intercal x

        Args:
            x: input data tensor :math:`X`
        """

    @abstractmethod
    def from_latent_to_output(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Evaluate the mapping from the latent space to output.

        .. math::
            y = g(z)

        Args:
            z: latent space representation
        """

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # pylint: disable=arguments-differ
        r"""data :math:`X` :math:`\rightarrow` output :math:`Y`.

        .. math::
            y = g(\phi(x))

        Args:
            x: input data tensor :math:`X`
        """
        return self.from_latent_to_output(self.to_latent(x, **kwargs), **kwargs)

    @abstractmethod
    def fix_gauge(self, gauge_mask: torch.Tensor):
        """Perform gauge-fixing procedure on latent space parameters.

        Args:
            gauge_mask: 0/1 mask array the same shape as latent space input with 1s
                        for parameters that should be zeroed
        """

    @property
    def sequence_length(self):
        r"""input amino acid sequence length"""
        alphabet_length = len(self.alphabet)
        assert self.input_size % alphabet_length == 0
        return self.input_size // alphabet_length

    def numpy_single_mutant_predictions(self) -> np.ndarray:
        r"""Single mutant predictions as a numpy array of shape (AAs,
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

    def single_mutant_predictions(self) -> List[pd.DataFrame]:
        r"""Single mutant predictions as a list (across output dimensions) of
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

    def monotonic_params_from_latent_space(self) -> Generator[torch.Tensor, None, None]:
        r"""Yields parameters to be floored to zero in a monotonic
        model. This is every parameter after the latent space excluding bias
        parameters.

        We follow the convention that the latent layers of a network are
        named like ``latent_layer*`` and the weights and bias are denoted
        ``layer_name*.weight`` and ``layer_name*.bias``.

        Layers from nested modules will be prefixed (e.g. with Independent).
        """
        for name, param in self.named_parameters():
            parse_name = name.split(".")
            # NOTE: minus indices allow for prefixes from nested module params
            is_latent_layer = parse_name[-2] == "latent_layer"
            is_bias = parse_name[-1] == "bias"
            if not is_latent_layer and not is_bias:
                yield param

    def _reflect_monotonic_params(self):
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
            getattr(self, layer_name).reset_parameters()

        if self.monotonic_sign is not None:
            self._reflect_monotonic_params()

    def set_require_grad_for_all_parameters(self, value: bool):
        r"""Set ``require_grad`` for all parameters.

        Args:
            value: ``require_grad=value`` for all parameters
        """
        for param in self.parameters():
            param.requires_grad = value

    def _default_training_style(self):
        """Set the training style to default."""
        click.echo("Training in default style.")
        self.set_require_grad_for_all_parameters(True)

    def seq_to_binary(self, seq):
        """Takes a string of amino acids and creates an appropriate one-hot
        encoding."""
        # Get indices to place 1s for present amino acids.
        assert self.input_size == len(seq) * len(
            self.alphabet
        ), "Sequence size doesn't match training sequences."
        assert set(seq).issubset(
            self.alphabet
        ), "Input sequence has character(s) outside of model's alphabet."
        alphabet_dict = {letter: idx for idx, letter in enumerate(self.alphabet)}
        amino_acid_idx = [alphabet_dict[aa] for aa in seq]
        indices = torch.zeros(len(seq), dtype=torch.long, requires_grad=False)
        for site, _ in enumerate(seq):
            indices[site] = (site * len(self.alphabet)) + amino_acid_idx[site]

        # Generate encoding.
        encoding = torch.zeros((1, len(seq) * len(self.alphabet)), requires_grad=False)
        encoding[0, indices.data] = 1.0

        return encoding[0]


class LinearModel(TorchdmsModel):
    r"""A linear model, expressed as a single layer neural network with no nonlinear activations.

    Args:
        args: positional arguments, see :py:class:`torchdms.model.TorchdmsModel`
        kwargs: keyword arguments, see :py:class:`torchdms.model.TorchdmsModel`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Note: to keep the WT sequence at the origin, we have zero bias, and
        # then a separate layer to store the bias term
        self.latent_layer = nn.Linear(self.input_size, self.output_size, bias=False)
        self.wt_activity = nn.Parameter(torch.zeros(self.output_size))
        self.layers = ["latent_layer"]

    @property
    def characteristics(self) -> Dict:
        return {}

    @property
    def latent_dim(self) -> int:
        return 1

    def beta_coefficients(self) -> torch.Tensor:
        return self.latent_layer.weights.data

    def from_latent_to_output(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.latent_layer(z + self.wt_activity)

    def regularization_loss(self) -> torch.Tensor:
        return 0.0

    def str_summary(self) -> str:
        return "Linear"

    def to_latent(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.latent_layer(x)

    def fix_gauge(self, gauge_mask):
        pass


class Escape(TorchdmsModel):
    r"""A model of viral escape with multiple epitopes, each with its own latent layer.

    The output is assumed to be escape fraction :math:`\in (0, 1)`. The nonlinearity is
    fixed as the product of Hill functions for each epitope.

    .. todo::
        model math markup, and schematic needed here

    Args:
        num_epitopes: number of epitopes (latent dimensions)
        args: base positional arguments, see :py:class:`torchdms.model.TorchdmsModel`
        beta_l1_coefficient: lasso penalty on latent space :math:`beta` coefficients
        kwargs: base keyword arguments, see :py:class:`torchdms.model.TorchdmsModel`

    """

    def __init__(self, num_epitopes, *args, beta_l1_coefficient: float = 0, **kwargs):
        super().__init__(*args, **kwargs)

        # set model attributes
        self.num_epitopes = num_epitopes
        self.beta_l1_coefficient = beta_l1_coefficient

        for i in range(self.num_epitopes):
            # NOTE: we track the wt activity as a separate layer, rather than
            # a bias term in the latent layer, so that the wt sequences is at
            # the origin in the latent space, like other torchdms models.
            setattr(
                self, f"latent_layer_epi{i}", nn.Linear(self.input_size, 1, bias=False)
            )
            setattr(self, f"wt_activity_epi{i}", nn.Parameter(torch.zeros(1)))

    @property
    def characteristics(self) -> Dict:
        return {
            "num_epitopes": self.num_epitopes,
            "beta_l1_coefficient": self.beta_l1_coefficient,
        }

    @property
    def latent_dim(self) -> int:
        return self.num_epitopes

    def str_summary(self):
        return "Escape"

    def to_latent(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.cat(
            tuple(
                getattr(self, f"latent_layer_epi{i}")(x)
                for i in range(self.num_epitopes)
            ),
            dim=1,
        )

    def wt_activity(self) -> torch.Tensor:
        r"""Wild type activity values for each epitope"""
        return torch.cat(
            tuple(getattr(self, f"wt_activity_epi{i}") for i in range(self.num_epitopes))
        )

    def from_latent_to_output(
        self, z: torch.Tensor, concentrations: torch.Tensor = None
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        log_concentrations = (
            0 if concentrations is None else torch.log(concentrations.unsqueeze(1))
        )
        escape_fractions = torch.sigmoid(z + self.wt_activity() - log_concentrations)
        return torch.unsqueeze(torch.prod(escape_fractions, 1), 1)

    def beta_coefficients(self):
        beta_coefficients_data = torch.cat(
            tuple(
                getattr(self, f"latent_layer_epi{i}").weight.data
                for i in range(self.num_epitopes)
            )
        )
        return beta_coefficients_data[:, : self.input_size]

    def regularization_loss(self):
        """Lasso penalty of single mutant effects, a scalar-valued
        :py:class:`torch.Tensor`"""
        penalty = self.beta_l1_coefficient * sum(
            getattr(self, f"latent_layer_epi{i}").weight.norm(1)
            for i in range(self.num_epitopes)
        )
        return penalty

    def fix_gauge(self, gauge_mask):
        self.beta_coefficients()[gauge_mask] = 0


class FullyConnected(TorchdmsModel):
    r"""A flexible fully connected neural network model.

    .. todo::
        schematic image

    Args:
        layer_sizes: Sequence of widths for each layer between input and output.
        activations: Corresponding activation functions for each layer.
                     The first layer with ``None`` or ``nn.Identity()`` activation
                     is the latent space.

                     .. todo::
                         allowable activation function names?

        args: base positional arguments, see :py:class:`torchdms.model.TorchdmsModel`
        beta_l1_coefficient: lasso penalty on latent space :math:`\beta` coefficients
        interaction_l1_coefficient: lasso penalty on parameters in the pre-latent
                                    interaction layer(s)
        kwargs: base keyword arguments, see :py:class:`torchdms.model.TorchdmsModel`

    Example:

        With ``layer_sizes = [10, 2, 10, 10]`` and
        ``activations = [nn.ReLU(), None, nn.ReLU(), nn.ReLU()]``
        we have a latent space of 2 nodes, feeding into
        two more dense layers, each with 10 nodes, before the output.
        Layers before the latent layer are a nonlinear module for site-wise
        interactions, in this case one layer of 10 nodes.
        The latent space has skip connections directly from the input layer,
        so we always model single mutation effects.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[Callable],
        *args,
        beta_l1_coefficient: float = 0.0,
        interaction_l1_coefficient: float = 0.0,
        **kwargs,
    ):
        if not len(layer_sizes) == len(activations):
            raise ValueError(
                f"{len(layer_sizes)} layer sizes inconsistent with {len(activations)} activations"
            )
        for idx, activation in enumerate(activations):
            if activation is None:
                activations[idx] = nn.Identity()
            elif not callable(activation):
                raise ValueError(f"activation function {activation} is not recognized")

        super().__init__(*args, **kwargs)
        self.layers = []
        self.activations = activations
        self.beta_l1_coefficient = beta_l1_coefficient
        self.interaction_l1_coefficient = interaction_l1_coefficient

        try:
            self.latent_idx = [type(activation) for activation in self.activations].index(nn.Identity)
        except ValueError:
            self.latent_idx = 0

        # additive model
        if len(layer_sizes) == 0:
            layer_name = "latent_layer"
            self.layers.append(layer_name)
            setattr(self, layer_name, nn.Linear(self.input_size, self.output_size))

        # all other models
        else:
            input_size = self.input_size
            prefix = "interaction"
            # the pre-latent interaction network should have zero bias, as well as
            # the latent layer, so that WT is at the origin
            bias = False
            for layer_index, num_nodes in enumerate(layer_sizes):
                if layer_index == self.latent_idx:
                    layer_name = "latent_layer"
                    if layer_index > 0:
                        # skip connection
                        input_size += self.input_size
                    prefix = "nonlinearity"
                else:
                    layer_name = f"{prefix}_{layer_index}"

                self.layers.append(layer_name)
                setattr(self, layer_name, nn.Linear(input_size, num_nodes, bias=bias))
                input_size = layer_sizes[layer_index]

                # Location parameter(s) for WT sequence are learned in the nonlinearity
                if prefix == "nonlinearity":
                    bias = True

            # final layer
            layer_name = "output_layer"
            self.layers.append(layer_name)
            setattr(self, layer_name, nn.Linear(layer_sizes[-1], self.output_size))

        if self.monotonic_sign is not None:
            # If monotonic, we want to initialize all parameters
            # which will be floored at 0, to a value above zero.
            self._reflect_monotonic_params()

    @property
    def characteristics(self) -> Dict:
        return {
            "activations": str(self.activations),
            "monotonic": self.monotonic_sign,
            "beta_l1_coefficient": self.beta_l1_coefficient,
            "interaction_l1_coefficient": self.interaction_l1_coefficient,
        }

    @property
    def internal_layer_dimensions(self) -> List[int]:
        """List of widths of internal layers."""
        return [getattr(self, layer).out_features for layer in self.layers[:-1]]

    @property
    def is_linear(self) -> bool:
        """is this a linear model (no internal layers)?"""
        return len(self.internal_layer_dimensions) == 0

    @property
    def latent_dim(self) -> int:
        dims = self.internal_layer_dimensions
        if len(dims) == 0:
            return 0
        # else:
        return dims[self.latent_idx]

    def str_summary(self) -> str:
        if self.is_linear:
            return "linear"
        # else:
        if self.monotonic_sign is None:
            monotonic = "non-mono"
        else:
            monotonic = "mono"
        return f"{monotonic};" + ";".join(
            [
                f"{dim};{name}"
                for dim, name in zip(self.internal_layer_dimensions, self.activations)
            ]
        )

    def fix_gauge(self, gauge_mask):
        # use mask to zero out coefficients
        self.beta_coefficients()[gauge_mask] = 0
        # project remaining nonzero coefficients such that mean is -1
        for latent_dim in range(self.latent_dim):
            beta_vec = self.beta_coefficients()[latent_dim, ~gauge_mask[0]]
            self.beta_coefficients()[latent_dim, ~gauge_mask[0]] = (
                beta_vec - beta_vec.sum() / beta_vec.shape[0] - 1
            )

    def to_latent(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
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

    def from_latent_to_output(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.is_linear:
            return z
        # else:
        out = z
        for layer_name, activation in zip(
            self.layers[self.latent_idx + 1 : -1],
            self.activations[self.latent_idx + 1 :],
        ):
            print(getattr(self, layer_name)(out))
            out = activation(getattr(self, layer_name)(out))
        # The last layer acts without an activation, which is on purpose because we
        # don't want to be limited to the range of the activation.
        out = getattr(self, self.layers[-1])(out)
        if self.monotonic_sign:
            out *= self.monotonic_sign
        return out

    def beta_coefficients(self) -> torch.Tensor:
        """Beta coefficients (single mutant effects only, no interaction
        terms)"""
        # This implementation assumes the single mutant terms are indexed first
        return self.latent_layer.weight.data[:, : self.input_size]

    def regularization_loss(self) -> torch.Tensor:
        """Lasso penalty of single mutant effects and pre-latent interaction
        weights, a scalar-valued :py:class:`torch.Tensor`"""
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


class Independent(TorchdmsModel):
    r"""Parallel and independent submodules of type :py:class:`torchdms.model.FullyConnected`
    for each of two output dimensions.

    .. todo::
        schematic image

    Args:
        layer_sizes: Sequence of widths for each layer between input and output *for both
                     submodules*.
        activations: Corresponding activation functions for each layer *for both submodules*.
                     The first layer with ``None`` or ``nn.Identity()`` activation is the
                     latent space.

                     .. todo::
                         allowable activation function names?

        args: base positional arguments, see :py:class:`torchdms.model.TorchdmsModel`
        beta_l1_coefficients: lasso penalties on latent space :math:`\beta` coefficients
                              *for each submodule*
        interaction_l1_coefficients: lasso penalty on parameters in the pre-latent
                                    interaction layer(s) *for each submodule*
        kwargs: base keyword arguments, see :py:class:`torchdms.model.TorchdmsModel`
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[Callable],
        *args,
        beta_l1_coefficients: Tuple[float] = (0.0, 0.0),
        interaction_l1_coefficients: Tuple[float] = (0.0, 0.0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

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
            beta_l1_coefficients = (0.0, 0.0)
        if interaction_l1_coefficients is None:
            interaction_l1_coefficients = (0.0, 0.0)

        for i, model in enumerate(("bind", "stab")):
            self.add_module(
                f"model_{model}",
                FullyConnected(
                    layer_sizes,
                    activations,
                    self.input_size,
                    [self.target_names[i]],
                    self.alphabet,
                    beta_l1_coefficient=beta_l1_coefficients[i],
                    interaction_l1_coefficient=interaction_l1_coefficients[i],
                    **kwargs,
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
    def characteristics(self) -> Dict:
        return dict(
            [("bind_" + k, v) for k, v in self.model_bind.characteristics.items()]
            + [("stab_" + k, v) for k, v in self.model_stab.characteristics.items()]
        )

    @property
    def internal_layer_dimensions(self) -> List[int]:
        return self.model_bind.internal_layer_dimensions

    @property
    def is_linear(self) -> bool:
        return self.internal_layer_dimensions.is_linear

    @property
    def latent_dim(self) -> int:
        return self.model_bind.latent_dim + self.model_stab.latent_dim

    def str_summary(self) -> str:
        return (
            f"{self.__class__.__name__}: ({self.model_bind.str_summary()}, "
            f"{self.model_stab.str_summary()})"
        )

    def from_latent_to_output(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.cat(
            (
                self.model_bind.from_latent_to_output(z[:, 0, None]),
                self.model_stab.from_latent_to_output(z[:, 1, None]),
            ),
            1,
        )

    def to_latent(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.cat(
            (self.model_bind.to_latent(x), self.model_stab.to_latent(x)), 1
        )

    def beta_coefficients(self) -> torch.Tensor:
        """beta coefficients, skip coefficients only if interaction module."""
        beta_coefficients_data = torch.cat(
            (
                self.model_bind.latent_layer.weight.data,
                self.model_stab.latent_layer.weight.data,
            )
        )
        return beta_coefficients_data[:, : self.input_size]

    def regularization_loss(self) -> torch.Tensor:
        return (
            self.model_bind.regularization_loss()
            + self.model_stab.regularization_loss()
        )

    def fix_gauge(self, gauge_mask):
        self.model_bind.fix_gauge(torch.unsqueeze(gauge_mask[0], dim=0))
        self.model_stab.fix_gauge(torch.unsqueeze(gauge_mask[1], dim=0))


class Conditional(Independent):
    r"""Allows the latent space for the second output feature (i.e. stability)
    to feed forward into the network for the first output feature (i.e.
    binding).

    This requires a one-dimensional latent space (see
    :py:meth:`torchdms.model.Conditional.from_latent_to_output` to see why).

    .. todo::
        schematic image:
        https://user-images.githubusercontent.com/1173298/89943302-d4524d00-dbd2-11ea-827d-6ad6c238ff52.png

    Args:
        args: positional arguments, see :py:class:`torchdms.model.Independent`
        kwargs: keyword arguments, see :py:class:`torchdms.model.Independent`
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

    def from_latent_to_output(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.cat(
            (
                # The nonlinearity for the binding output gets to see all latent
                # dimensions.
                self.model_bind.from_latent_to_output(z),
                # The nonlinearity for the stability output only sees the latent
                # information from the stability part of the model.
                self.model_stab.from_latent_to_output(z[:, 1, None]),
            ),
            1,
        )


class ConditionalSequential(Conditional):
    r"""Conditional with sequential training: stability then binding.

    .. todo::
        schematic image

    Args:
        args: positional arguments, see :py:class:`torchdms.model.Conditional`
        kwargs: keyword arguments, see :py:class:`torchdms.model.Conditional`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_style_sequence = [
            self._only_train_stab_style,
            self._only_train_bind_style,
        ]

    def _only_train_bind_style(self):
        click.echo("Only training bind.")
        self.set_require_grad_for_all_parameters(True)
        self.model_stab.set_require_grad_for_all_parameters(False)

    def _only_train_stab_style(self):
        click.echo("Only training stab.")
        self.set_require_grad_for_all_parameters(True)
        self.model_bind.set_require_grad_for_all_parameters(False)
