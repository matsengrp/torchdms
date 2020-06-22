"""Our models."""
import re
import click
import torch
import torch.nn as nn
from torchdms.utils import from_pickle_file


def identity(x):
    """The identity function, to be used as "no activation."."""
    return x


class VanillaGGE(nn.Module):
    """Make it just how you like it.

    input size can be inferred for the train/test datasets
    and output can be inferred from the number of targets
    specified. the rest should simply be fed in as a list
    like:

    layer_sizes = [2, 10, 10]

    means we have a 'latent' space of 2 nodes, connected to
    two more dense layers, each with 10 nodes, before the output.

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
        output_size,
        monotonic_sign=None,
        beta_l1_coefficient=0.0,
    ):
        super(VanillaGGE, self).__init__()
        self.monotonic_sign = monotonic_sign
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        self.activations = activations
        self.beta_l1_coefficient = beta_l1_coefficient

        assert len(layer_sizes) == len(activations)

        layer_name = "input_layer"

        # additive model
        if len(layer_sizes) == 0:
            self.layers.append(layer_name)
            setattr(self, layer_name, nn.Linear(input_size, output_size))

        # all other models
        else:
            in_size = input_size
            bias = False
            for layer_index, num_nodes in enumerate(layer_sizes):
                self.layers.append(layer_name)
                setattr(self, layer_name, nn.Linear(in_size, num_nodes, bias=bias))
                layer_name = f"internal_layer_{layer_index}"
                in_size = layer_sizes[layer_index]
                bias = True

            # final layer
            layer_name = "output_layer"
            self.layers.append(layer_name)
            setattr(self, layer_name, nn.Linear(layer_sizes[-1], output_size))

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
        }

    @property
    def internal_layer_dimensions(self):
        return [
            getattr(self, layer).in_features
            for layer in self.layers
            if "input" not in layer
        ]

    @property
    def is_linear(self):
        return len(self.internal_layer_dimensions) == 0

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

    def __str__(self):
        return super(VanillaGGE, self).__str__() + "\n" + self.characteristics.__str__()

    def forward_of_layers_and_activations(self, layers, activations, x):
        out = x
        for layer_name, activation in zip(layers[:-1], activations):
            out = activation(getattr(self, layer_name)(out))
        # The last layer acts without an activation, which is on purpose because we
        # don't want to be limited to the range of the activation.
        out = getattr(self, layers[-1])(out)
        if self.monotonic_sign:
            out *= self.monotonic_sign
        return out

    def forward(self, x):  # pylint: disable=arguments-differ
        return self.forward_of_layers_and_activations(self.layers, self.activations, x)

    def from_latent(self, x):
        """Evaluate the mapping from the latent space to output."""
        if self.is_linear:
            return x
        # else:
        out = self.activations[0](x)
        return self.forward_of_layers_and_activations(
            self.layers[1:], self.activations[1:], out
        )

    def to_latent(self, x):
        """Map input into latent space."""
        return getattr(self, self.layers[0])(x)

    def regularization_loss(self):
        """L1-penalize betas for all latent space dimensions except for the
        first one."""
        if self.beta_l1_coefficient == 0.0:
            return 0.0
        beta_parameters = next(self.parameters())
        # The dimensions of the latent space are laid out as rows of the parameter
        # matrix.
        latent_space_dim = beta_parameters.shape[0]
        if latent_space_dim == 1:
            return 0.0
        # else:
        return self.beta_l1_coefficient * torch.sum(
            torch.abs(
                beta_parameters.narrow(
                    0,  # Slice along the 0th dimension.
                    1,  # Start penalizing after the first dimension.
                    latent_space_dim - 1,  # Penalize all subsequent dimensions.
                )
            )
        )

    def reflect_monotonic_params(self):
        """Ensure that monotonic parameters are non-negative by flipping their
        sign."""
        for param in monotonic_params_from_latent_space(self):
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
            self.reflect_monotonic_params()


def monotonic_params_from_latent_space(model: VanillaGGE):
    """following the hueristic that the input layer of a network is named
    'input_layer' and the weight bias are denoted:

    layer_name.weight
    layer_name.bias.

    this function returns all the parameters
    to be floored to zero in a monotonic model.
    this is every parameter after the latent space
    excluding bias parameters.
    """
    for name, param in model.named_parameters():
        parse_name = name.split(".")
        is_input_layer = parse_name[0] == "input_layer"
        is_bias = parse_name[1] == "bias"
        if not is_input_layer and not is_bias:
            yield param


KNOWN_MODELS = {
    "VanillaGGE": VanillaGGE,
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


def model_of_string(model_string, data_path, monotonic_sign):
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
    if model_name == "VanillaGGE":
        if len(layers) == 0:
            click.echo("LOG: No layers provided, so I'm creating a linear model.")
        for layer in layers:
            if not isinstance(layer, int):
                raise TypeError("All layer input must be integers")
        model = VanillaGGE(
            test_dataset.feature_count(),
            layers,
            activations,
            test_dataset.targets.shape[1],
            monotonic_sign=monotonic_sign,
        )
    else:
        model = KNOWN_MODELS[model_name](test_dataset.feature_count())
    return model
