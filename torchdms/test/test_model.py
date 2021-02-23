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

    beta = model.latent_layer.weight[:, : model.input_size]
    grad_beta = model.latent_layer.weight.grad[:, : model.input_size]

    assert torch.equal(grad_beta, model.beta_l1_coefficient * beta.sign())
