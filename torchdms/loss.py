"""Loss functions."""
import torch


def l1(y_true, y_predicted, loss_decay=None, exp_target=None):
    """Mean square error, perhaps with loss decay."""
    y_true_squoze = y_true.squeeze()
    y_predicted_squoze = y_predicted.squeeze()
    if loss_decay:
        return torch.sum(
            torch.exp(loss_decay * y_true_squoze)
            * (y_true_squoze - y_predicted_squoze).abs()
        )
    if exp_target:
        y_true_squoze = torch.pow(exp_target, y_true.squeeze())
        y_predicted_squoze = torch.pow(exp_target, y_predicted.squeeze())
    # else:
    return torch.nn.functional.l1_loss(y_true_squoze, y_predicted_squoze)


def mse(y_true, y_predicted, loss_decay=None, exp_target=None):
    """Mean square error, perhaps with loss decay."""
    y_true_squoze = y_true.squeeze()
    y_predicted_squoze = y_predicted.squeeze()
    if loss_decay:
        return torch.sum(
            torch.exp(loss_decay * y_true_squoze)
            * (y_true_squoze - y_predicted_squoze) ** 2
        )
    if exp_target:
        y_true_squoze = torch.pow(exp_target, y_true.squeeze())
        y_predicted_squoze = torch.pow(exp_target, y_predicted.squeeze())
        return torch.nn.functional.mse_loss(y_true_squoze, y_predicted_squoze)
    # else:
    return torch.nn.functional.mse_loss(y_true.squeeze(), y_predicted.squeeze())


def rmse(y_true, y_predicted, loss_decay=None):
    """Root mean square error, perhaps with loss decay."""
    return mse(y_true, y_predicted, loss_decay).sqrt()
