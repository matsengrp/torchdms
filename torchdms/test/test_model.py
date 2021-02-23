"""
Testing model module
"""
import numpy as np
import torch
from torchdms.model import FullyConnected, identity


def test_regularization_loss():
    """Test regularization loss with l1 penalty on beta coefficients."""

    model = FullyConnected(10, [10], [identity], ["foo"], "ABCD",
            beta_l1_coefficient=2.0, interaction_l1_coefficient=1.5)

    model.zero_grad()

    loss = model.regularization_loss()
    print(loss)
    print(loss.grad)

    print(model.beta_l1_coefficient)

    # assert torch.allclose(torch.from_numpy(approx_true), approx_est, rtol=0.001)
