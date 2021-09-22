"""Loss functions and functions relevant to losses."""
import torch


def weighted_loss(base_loss):
    """Generic loss function decorator with loss decay or target
    exponentiation."""

    def wrapper(y_true, y_predicted, loss_decay=None, exp_target=None):
        if exp_target:
            y_true = (torch.pow(exp_target, y_true),)
            y_predicted = (torch.pow(exp_target, y_predicted),)
        if loss_decay:
            weights = torch.exp(loss_decay * y_true)
            sample_losses = base_loss(y_true, y_predicted, reduction="none")
            return torch.mean(weights * sample_losses)
        # else:
        return base_loss(y_true, y_predicted)

    wrapper.__doc__ = (
        base_loss.__name__ + """, perhaps with loss decay or target exponentiation"""
    )
    return wrapper


l1 = weighted_loss(torch.nn.functional.l1_loss)

mse = weighted_loss(torch.nn.functional.mse_loss)


def rmse(*args, **kwargs):
    """Root mean square error, perhaps with loss decay or target
    exponentiation."""
    return mse(*args, **kwargs).sqrt()


def sitewise_group_lasso(matrix):
    """The sum of the 2-norm across columns.

    We omit the square root of the group sizes, as they are all constant
    in our case.
    """
    assert len(matrix.shape) == 2
    return torch.norm(matrix, p=2, dim=0).sum()


def product_penalty(betas):
    """Computes l1 norm of product of betas across latent dimensions."""
    assert len(betas.shape) == 3
    return torch.prod(betas, 0).norm(1)


def diff_penalty(betas):
    """Computes l1 norm of the difference between adjacent betas for each
    latent dimension."""
    penalty = torch.zeros(1)
    for i in range(betas.size()[0]):
        penalty += (betas[i][1:] - betas[i][:-1]).norm(1)
    return penalty


def sum_diff_penalty(betas):
    """Computes l1 norm of the difference between aggregated betas at adjacent
    sites for each latent dimension."""
    penalty = torch.zeros(1)
    site_sums = torch.sum(
        torch.abs(betas.view(betas.size()[0], int(betas.size()[1] / 21), 21)), 2
    )
    for i in range(betas.size()[0]):
        penalty += (site_sums[i][1:] - site_sums[i][:-1]).norm(1)
    return penalty
