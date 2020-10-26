"""Loss functions."""
import torch


def l1(y_true, y_predicted, loss_decay=None, exp_target=None):
    """Mean square error, perhaps with loss decay."""
    if loss_decay:
        y_true_squoze = y_true.squeeze()
        return torch.sum(
            torch.exp(loss_decay * y_true_squoze)
            * (y_true_squoze - y_predicted.squeeze()).abs()
        )
    elif exp_target:
        y_true_squoze = torch.pow(exp_target,y_true.squeeze())
        y_predicted_squoze = torch.pow(exp_target, y_predicted.squeeze())
        return torch.nn.functional.mse_loss(y_true_exp, y_predicted_exp)
    # else:
    return torch.nn.functional.l1_loss(y_true.squeeze(), y_predicted.squeeze())


def mse(y_true, y_predicted, loss_decay=None, exp_target=None):
    """Mean square error, perhaps with loss decay."""
    if loss_decay:
        y_true_squoze = y_true.squeeze()
        return torch.sum(
            torch.exp(loss_decay * y_true_squoze)
            * (y_true_squoze - y_predicted.squeeze()) ** 2
        )
    elif exp_target:
        y_true_squoze = torch.pow(exp_target,y_true.squeeze())
        y_predicted_squoze = torch.pow(exp_target, y_predicted.squeeze())
        return torch.nn.functional.mse_loss(y_true_exp, y_predicted_exp)
    # else:
    return torch.nn.functional.mse_loss(y_true.squeeze(), y_predicted.squeeze())


def rmse(y_true, y_predicted, loss_decay=None, exp_target=None):
    """Root mean square error, perhaps with loss decay."""
    return mse(y_true, y_predicted, loss_decay).sqrt()
