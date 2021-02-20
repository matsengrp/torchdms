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


def l1_epitope_product(betas):
    """Computes l1 norm of product of betas across epitopes."""
    return torch.prod(betas, 0).norm(1)


def distance_penalty(betas, num_aa, num_sites, num_nbrs):
    """Computes a distance-based penalty score.

    Prioritizes having large betas closer together in distance.
    """
    ranked_betas = torch.argsort(betas, descending=True)
    ranked_betas_by_site = ranked_betas / num_aa  # index by site instead of site * AA
    top_betas = ranked_betas_by_site.index_select(1, torch.arange(0, num_nbrs))

    penalty = 0
    for i in range(betas.size()[0]):
        for idx1, j in enumerate(top_betas[i]):
            for idx2, k in enumerate(top_betas[i]):
                if idx1 != idx2:
                    penalty += (j - k) ** 2
    return penalty
