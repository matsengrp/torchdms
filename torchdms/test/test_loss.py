"""
Testing for loss.py.
"""
from math import exp
from torch import Tensor
from torchdms.loss import l1, mse, l1_penalty, product_penalty, diff_penalty


def test_mse_loss():
    """Test mean squared error loss with loss decay."""
    y_true = Tensor([0.1, 4.0])
    y_predicted = Tensor([0.2, 9.0])
    correct_decayed_loss = (
        exp(0.3 * 0.1) * (0.1 - 0.2) ** 2 + exp(0.3 * 4.0) * (4.0 - 9.0) ** 2
    )
    assert mse(y_true, y_predicted, 0.3) == correct_decayed_loss


def test_l1_loss():
    """Test l1 loss with loss decay."""
    y_true = Tensor([0.1, 4.0])
    y_predicted = Tensor([0.2, 9.0])
    correct_decayed_loss = exp(0.3 * 0.1) * (0.2 - 0.1) + exp(0.3 * 4.0) * (9.0 - 4.0)
    assert l1(y_true, y_predicted, 0.3) == correct_decayed_loss


def test_l1_penalty():
    """Test l1 norm of betas"""
    betas = Tensor([[0.1, 2, 5], [1, 0, -0.5]])
    correct_sum = abs(0.1) + abs(2) + abs(5) + abs(1) + abs(0) + abs(-0.5)
    assert l1_penalty(betas) == correct_sum


def test_product_penalty():
    """Test l1 norm of product of betas across latent dimensions."""
    betas = Tensor([[0.1, 2, 5], [1, 0, -0.5]])
    correct_sum = abs(0.1 * 1) + abs(2 * 0) + abs(5 * -0.5)
    assert product_penalty(betas) == correct_sum


def test_diff_penalty():
    """Test l1 norm of difference between adjacent betas across latent dimensions"""
    betas = Tensor([[0.1, 2, 5], [1, 0, -0.5]])
    correct_sum = abs(2 - 0.1) + abs(5 - 2) + abs(0 - 1) + abs(-0.5 - 0)
    assert diff_penalty(betas) == correct_sum
