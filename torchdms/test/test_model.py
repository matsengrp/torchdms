"""
Testing model module
"""
from torchdms.model import FullyConnected


def test_regularization_loss():
    """Test regularization loss with l1 penalty on beta coefficients."""

    model = FullyConnected(10, [10], [None], [None], None,
                           beta_l1_coefficient=2.0,
                           interaction_l1_coefficient=1.5)

    model.zero_grad()
    loss = model.regularization_loss()
    loss.backward()

    print(model.latent_layer.weight.grad)

    print(model.beta_l1_coefficient)

    # assert torch.allclose(torch.from_numpy(approx_true), approx_est, rtol=0.001)
