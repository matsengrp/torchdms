"""Loss functions and functions relevant to losses."""
import torch


def make_squoze_y_pair(y_true, y_predicted, exp_target):
    """Make squeezed versions of y_true and y_predicted, perhaps using
    exp_target as a base for exponentiation."""
    if exp_target is not None:
        return (
            torch.pow(exp_target, y_true.squeeze()),
            torch.pow(exp_target, y_predicted.squeeze()),
        )
    # else:
    return y_true.squeeze(), y_predicted.squeeze()


def l1(y_true, y_predicted, loss_decay=None, exp_target=None):
    """Mean square error, perhaps with loss decay."""
    y_true_squoze, y_predicted_squoze = make_squoze_y_pair(
        y_true, y_predicted, exp_target
    )
    if loss_decay:
        return torch.sum(
            torch.exp(loss_decay * y_true_squoze)
            * (y_true_squoze - y_predicted_squoze).abs()
        )
    # else:
    return torch.nn.functional.l1_loss(y_true_squoze, y_predicted_squoze)


def mse(y_true, y_predicted, loss_decay=None, exp_target=None):
    """Mean square error, perhaps with loss decay."""
    y_true_squoze, y_predicted_squoze = make_squoze_y_pair(
        y_true, y_predicted, exp_target
    )
    if loss_decay:
        return torch.sum(
            torch.exp(loss_decay * y_true_squoze)
            * (y_true_squoze - y_predicted_squoze) ** 2
        )
    # else:
    return torch.nn.functional.mse_loss(y_true_squoze, y_predicted_squoze)


def rmse(y_true, y_predicted, loss_decay=None):
    """Root mean square error, perhaps with loss decay."""
    return mse(y_true, y_predicted, loss_decay).sqrt()


def group_lasso(matrix):
    """The sum of the 2-norm across columns."""
    shape = list(matrix.shape)
    assert len(shape) == 2
    total = torch.Tensor([0.0])
    for i in range(shape[1]):
        total += matrix[:, i].norm(2)
    return total
