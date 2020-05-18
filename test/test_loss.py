"""
Testing for loss.py.
"""
from math import exp
from torch import Tensor
from torchdms.loss import mse


def test_mse_loss():
    """Test mean squared error loss with loss decay."""
    y_true = Tensor([0.1, 4.0])
    y_predicted = Tensor([0.2, 9.0])
    correct_decayed_loss = (
        exp(0.3 * 0.1) * (0.1 - 0.2) ** 2 + exp(0.3 * 4.0) * (4.0 - 9.0) ** 2
    )
    assert mse(y_true, y_predicted, 0.3) == correct_decayed_loss
