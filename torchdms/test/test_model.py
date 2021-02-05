### THIS SCRIPT WAS ADDED BY TIM =====================
import pathlib
import json
import os
import random
import click
import click_config_file
import pandas as pd
import torch
import torchdms
from torchdms.analysis import Analysis
from torchdms.data import (
    check_onehot_encoding,
    partition,
    prep_by_stratum_and_export,
    SplitDataset,
    summarize_dms_variants_dataframe,
)
from torchdms.evaluation import (
    build_evaluation_dict,
    complete_error_summary,
    error_df_of_evaluation_dict,
)
from torchdms.loss import l1, mse, rmse
from torchdms.model import model_of_string
from torchdms.plot import (
    beta_coefficients,
    build_geplot_df,
    build_2d_nonlinearity_df,
    plot_error,
    plot_geplot,
    plot_2d_geplot,
    plot_heatmap,
    plot_svd,
    plot_test_correlation,
)
from torchdms.utils import (
    float_list_of_comma_separated_string,
    from_pickle_file,
    from_json_file,
    make_cartesian_product_hierarchy,
    to_pickle_file,
)


def set_random_seed(seed):
    if seed is not None:
        click.echo(f"LOG: Setting random seed to {seed}.")
        torch.manual_seed(seed)
        random.seed(seed)


def train(
    model_path,
    data_path,
    loss_fn,
    loss_weight_span,
    exp_target,
    batch_size,
    learning_rate,
    min_lr,
    patience,
    device,
    independent_starts,
    independent_start_epochs,
    simple_training,
    epochs,
    dry_run,
    seed,
    beta_rank,
):
    """Train a model, saving trained model to original location."""
    if dry_run:
        print_method_name_and_locals("train", locals())
        return
    set_random_seed(seed)

    model = torch.load(model_path)
    print(model)
    data = from_pickle_file(data_path)

    analysis_params = {
        "model": model,
        "model_path": model_path,
        "val_data": data.val,
        "train_data_list": data.train,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": device,
    }

    analysis = Analysis(**analysis_params)
    known_loss_fn = {"l1": l1, "mse": mse, "rmse": rmse}
    if loss_fn not in known_loss_fn:
        raise IOError(f"Loss function '{loss_fn}' not known.")
    loss_fn = known_loss_fn[loss_fn]

    if simple_training:
        click.echo(f"Starting simple training for {epochs} epochs.")
        analysis.simple_train(epochs, loss_fn)
        return
    if exp_target:
        click.echo(f"Exponentiating targets with base {exp_target}.")
        if loss_weight_span is not None:
            click.echo(
                "NOTE: you have indicated that you would like to exponentiate your targets "
                "while also using weight decay. Since these procedures are redundant, "
                "weight decay will be turned off."
            )
            loss_weight_span = None
    # else:
    if loss_weight_span is not None:
        click.echo(
            "NOTE: you are using loss decays, which assumes that you want to up-weight "
            "the loss function when the true target value is large. Is that true?"
        )
    if beta_rank is not None:
        click.echo(f"NOTE: Using rank-{beta_rank} approximation for beta coefficents.")
    training_params = {
        "independent_start_count": independent_starts,
        "independent_start_epoch_count": independent_start_epochs,
        "epoch_count": epochs,
        "loss_fn": loss_fn,
        "patience": patience,
        "min_lr": min_lr,
        "loss_weight_span": loss_weight_span,
        "exp_target": exp_target,
        "beta_rank": beta_rank,
    }
    click.echo(f"Starting training. {training_params}")
    analysis.multi_train(**training_params)


if __name__ == "__main__":

    params = {
        "model_path": "../map-multi-mutants/sim_01292021/escape.model",
        "data_path": "../map-multi-mutants/sim_01292021/prepped_libsim.pkl",
        "loss_fn": "l1",
        "loss_weight_span": None,
        "exp_target": None,
        "batch_size": 500,
        "learning_rate": 0.001,
        "min_lr": 1e-5,
        "patience": 10,
        "device": "cpu",
        "independent_starts": 0,
        "independent_start_epochs": 5,
        "simple_training": False,
        "epochs": 100,
        "dry_run": False,
        "seed": 0,
        "beta_rank": None,
    }
    train(**params)

# python3 torchdms/test/test_model.py
