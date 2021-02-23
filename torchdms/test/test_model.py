"""
Testing model module
"""
import torch
from torchdms.model import FullyConnected, identity


def test_regularization_loss():
    """Test regularization loss gradient."""

    model = FullyConnected(
        10,
        [10, 2],
        [None, identity],
        [None],
        None,
        beta_l1_coefficient=torch.rand(1),
        interaction_l1_coefficient=torch.rand(1),
    )

    loss = model.regularization_loss()
    loss.backward()

    for layer in model.layers:
        layer_type = layer.split("_")[0]
        if layer_type == "latent":
            weight = model.latent_layer.weight[:, : model.input_size]
            grad_weight = model.latent_layer.weight.grad[:, : model.input_size]
            penalty = model.beta_l1_coefficient
        elif layer_type == "interaction":
            weight = getattr(model, layer).weight
            grad_weight = getattr(model, layer).weight.grad
            penalty = model.interaction_l1_coefficient
        else:
            break
        assert torch.equal(grad_weight, penalty * weight.sign())
