"""
Testing model module
"""
import torch
from torchdms.model import FullyConnected


def test_regularization_loss():
    """Test regularization loss with l1 penalty on beta coefficients."""

    model = FullyConnected(10, [10], [None], [None], None,
                           beta_l1_coefficient=torch.rand(1))

    loss = model.regularization_loss()
    loss.backward()

    assert torch.equal(model.latent_layer.weight.grad,
                       model.beta_l1_coefficient * model.latent_layer.weight.sign())
