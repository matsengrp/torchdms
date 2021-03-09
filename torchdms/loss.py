"""Loss functions."""
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


def l1_penalty(betas):
    """textbook L1-regularization."""
    penalty = torch.zeros(1)
    for i in range(betas.size()[0]):
        penalty += torch.sum(torch.abs(betas[i]))
    return penalty


def product_penalty(betas):
    """Computes l1 norm of product of betas across latent dimensions."""
    return torch.sum(torch.abs(torch.prod(betas, 0)))


def diff_penalty(betas):
    """Computes l1 norm of the difference between adjacent betas for each
    latent dimension."""
    penalty = torch.zeros(1)
    for i in range(betas.size()[0]):
        penalty += torch.sum(torch.abs(betas[i][1:] - betas[i][:-1]))
    return penalty


def sum_diff_penalty(betas):
    """Computes l1 norm of the difference between aggregated betas at adjacent
    sites for each latent dimension."""
    penalty = torch.zeros(1)
    site_sums = torch.sum(
        torch.abs(betas.view(betas.size()[0], int(betas.size()[1] / 21), 21)), 2
    )
    for i in range(betas.size()[0]):
        penalty += torch.sum(torch.abs(site_sums[i][1:] - site_sums[i][:-1]))
    return penalty
