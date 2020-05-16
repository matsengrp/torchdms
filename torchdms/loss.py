"""Loss functions."""
import torch


def rmse(y_true, y_predicted, loss_decay=None):
    """Root mean square error, perhaps with loss decay."""
    return mse(y_true, y_predicted, loss_decay).sqrt()


def mse(y_true, y_predicted, loss_decay=None):
    """Mean square error, perhaps with loss decay."""
    # TODO make warning that good has to be big
    if loss_decay:
        y_true_squoze = y_true.squeeze()
        return torch.sum(
            torch.exp(loss_decay * y_true_squoze)
            * (y_true_squoze - y_predicted.squeeze()) ** 2
        )
    # else:
    return torch.nn.functional.mse_loss(y_true.squeeze(), y_predicted.squeeze())
