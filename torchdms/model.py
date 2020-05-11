import torch
import torch.nn as nn


class DMSFeedForwardModel(nn.Module):
    """
    Make it just how you like it.

    input size can be inferred for the train/test datasets
    and output can be inferred from the number of targets
    specified. the rest should simply be fed in as a list
    like:

    layers = [2, 10, 10]

    means we have a 'latent' space of 2 nodes, connected to
    two more dense layers, each with 10 layers, before the output.

    If layers is fed an empty list, the model will be a
    neural network equivilent of the Additive linear model.
    """

    def __init__(
        self,
        input_size,
        layers,
        output_size,
        activation_fn=torch.sigmoid,
        monotonic=False,
        beta_l1_coefficient=0.0,
    ):
        super(DMSFeedForwardModel, self).__init__()
        self.monotonic = monotonic
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        self.activation_fn = activation_fn
        self.beta_l1_coefficient = beta_l1_coefficient

        layer_name = f"input_layer"

        # additive model
        if len(layers) == 0:
            self.layers.append(layer_name)
            setattr(self, layer_name, nn.Linear(input_size, output_size))

        # all other models
        else:
            # all internal layers
            in_size = input_size
            bias = False
            for layer_index, num_nodes in enumerate(layers):
                self.layers.append(layer_name)
                setattr(self, layer_name, nn.Linear(in_size, num_nodes, bias=bias))
                layer_name = f"internal_layer_{layer_index}"
                in_size = layers[layer_index]
                bias = True

            # final layer
            layer_name = f"output_layer"
            self.layers.append(layer_name)
            setattr(self, layer_name, nn.Linear(num_nodes, output_size))

    def forward(self, x):
        out = x
        for layer_index in range(len(self.layers) - 1):
            out = self.activation_fn(getattr(self, self.layers[layer_index])(out))
        prediction = getattr(self, self.layers[-1])(out)
        return prediction

    def regularization_loss(self):
        """
        L1-penalize betas for all latent space dimensions except for the first one.
        """
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

    def from_latent(self, x):
        assert len(self.layers) != 0
        out = x
        for layer_index in range(1, len(self.layers) - 1):
            out = self.activation_fn(getattr(self, self.layers[layer_index])(out))
        prediction = getattr(self, self.layers[-1])(out)
        return prediction
