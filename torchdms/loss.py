import torch


def rmse(y, y_hat):
    return mse(y, y_hat).sqrt()


def mse(y, y_hat):
    return torch.nn.functional.mse_loss(y_hat.squeeze(), y.squeeze())
