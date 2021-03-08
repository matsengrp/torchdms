"""
Testing model module
"""
import torch
import torchdms.model
import torch.nn as nn


def test_latent_origin():
    """The WT sequence (zero tensor input) should lie at the origin of the latent space."""

    for model_name in torchdms.model.KNOWN_MODELS:

        input_size = 100
        base_args = [input_size, [None, None], None]

        activations = [nn.Identity(), nn.ReLU()]
        layer_sizes = [1, 10]

        if model_name == "Linear":
            args = base_args
        elif model_name == "Escape":
            num_epitopes = 2
            args = [num_epitopes, *base_args]
        elif model_name in (
            "FullyConnected",
            "Independent",
            "Conditional",
            "ConditionalSequential",
        ):
            args = [layer_sizes, activations, *base_args]
        else:
            raise NotImplementedError(model_name)

        model = torchdms.model.KNOWN_MODELS[model_name](*args)

        z_WT = model.to_latent(torch.zeros(1, input_size))

        assert torch.equal(z_WT, torch.zeros_like(z_WT))
