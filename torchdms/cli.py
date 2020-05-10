import click
import pandas as pd
import os
import torch

from torchdms.data import *
from torchdms.analysis import Analysis
from torchdms.model import *
from torchdms.loss import *
from torchdms.utils import *
from click import Choice, Path, command, group, option, argument


# Entry point
@group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """
    A generalized method to train neural networks
    on deep mutational scanning data.
    """
    pass


@cli.command(
    name="prep",
    short_help="Prepare a dataframe with aa subsitutions and targets in the \
    format needed to present to a neural network.",
)
@argument("in_path", required=True, type=click.Path(exists=True))
@argument("out_path", required=True, type=click.Path())
@argument("targets", type=str, nargs=-1, required=True)
@option(
    "--per-stratum-variants-for-test",
    type=int,
    required=False,
    default=100,
    show_default=True,
    help="This is the number samples for each stratum \
    to hold out for testing. \
    The rest of the examples will be used for training the model.",
)
@option(
    "--skip-stratum-if-count-is-smaller-than",
    type=int,
    required=False,
    default=250,
    show_default=True,
    help="If the total number of examples for any \
    particular stratum is lower than this number, \
    we throw out the stratum completely.",
)
def prep(
    in_path,
    out_path,
    targets,
    per_stratum_variants_for_test,
    skip_stratum_if_count_is_smaller_than,
):
    """
    Prepare data for training. IN_PATH should point to a pickle dump'd
    Pandas DataFrame containing the string encoded `aa_substitutions`
    column along with any TARGETS you specify. OUT_PATH is the
    location to dump the prepped data to another pickle file.
    """

    click.echo(f"LOG: Targets: {targets}")
    click.echo(f"LOG: Loading substitution data for: {in_path}")
    aa_func_scores, wtseq = from_pickle_file(in_path)
    click.echo(f"LOG: Successfully loaded data")

    total_variants = len(aa_func_scores.iloc[:, 1])
    click.echo(f"LOG: There are {total_variants} in this dataset")

    test_partition, partitioned_train_data = partition(
        aa_func_scores,
        per_stratum_variants_for_test,
        skip_stratum_if_count_is_smaller_than,
    )
    for train_part in partitioned_train_data:
        num_subs = len(train_part["aa_substitutions"][0].split())
        click.echo(
            f"LOG: There are {len(train_part)} training examples \
              for stratum: {num_subs}"
        )
    click.echo(f"LOG: There are {len(test_partition)} test points")
    click.echo(f"LOG: Successfully partitioned data")

    click.echo(f"LOG: preparing binary map dataset")
    to_pickle_file(
        prepare(test_partition, partitioned_train_data, wtseq, list(targets)), out_path
    )
    click.echo(
        f"LOG: Successfully finished prep and dumped BinaryMapDataset \
          object to {out_path}"
    )
    return None


@cli.command(name="create", short_help="Create a model")
@argument("data_path", type=click.Path(exists=True))
@argument("out_path", type=click.Path())
@argument("model_name")
@argument("layers", type=int, nargs=-1, required=False)
@option(
    "--monotonic",
    is_flag=True,
    help="If this flag is used, \
    then the model will be initialized with weights greater than zero. \
    During training with this model then, tdms will put a floor of \
    0 on all non-bias weights.",
)
def create(model_name, data_path, out_path, layers, monotonic):
    """
    Create a model. Model name can be the name of any
    of the functions defined in torch.models.

    If using the DMSFeedForwardModel model, you must provide some number of
    integer arguments following the model name to specify the number of nodes
    and layers for each (LAYERS argument).
    """
    known_models = {
        "DMSFeedForwardModel": DMSFeedForwardModel,
    }
    click.echo(f"LOG: searching for {model_name}")
    if model_name not in known_models:
        raise IOError(model_name + " not known")
    click.echo(f"LOG: found {model_name}")
    click.echo(f"LOG: loading training data")
    [test_BMD, _] = from_pickle_file(data_path)

    click.echo(f"LOG: Test data input size: {test_BMD.feature_count()}")
    click.echo(f"LOG: Test data output size: {test_BMD.targets.shape[1]}")
    if model_name == "DMSFeedForwardModel":
        if len(list(layers)) == 0:
            click.echo(f"LOG: No layers provided means creating a linear model")
        for layer in layers:
            if type(layer) != int:
                raise TypeError("All layer input must be integers")
        model = DMSFeedForwardModel(
            test_BMD.feature_count(), list(layers), test_BMD.targets.shape[1]
        )
    else:
        model = known_models[model_name](test_BMD.feature_count())
    click.echo(f"LOG: Successfully created model")

    # if monotonic, we want to initialize all parameters
    # which will be floored at 0, to a value above zero.
    if monotonic:
        click.echo(f"LOG: Successfully created model")

        # this flag will tell the ModelFitter to clamp (floor at 0)
        # the appropriate parameters after updating the weights
        model.monotonic = True
        for param in monotonic_params_from_latent_space(model):

            # https://pytorch.org/docs/stable/nn.html#linear
            # because the original distribution is
            # uniform between (-k, k) where k = 1/input_features,
            # we can simply transform all weights < 0 to their positive
            # counterpart
            param.data[param.data < 0] *= -1

    torch.save(model, out_path)
    click.echo(f"LOG: Model defined as: {model}")
    click.echo(f"LOG: Saved model to {out_path}")


# TODO train should just take the config file as a dict
@cli.command(name="train", short_help="Train a Model")
@argument("model_path", type=click.Path(exists=True))
@argument("data_path", type=click.Path(exists=True))
@option("--loss-out", type=click.Path(), required=False)
@option("--loss-fn", default="rmse", required=True, help="Loss function for training.")
@option(
    "--batch-size", default=500, show_default=True, help="Batch size for training.",
)
@option(
    "--learning-rate", default=1e-3, show_default=True, help="Initial learning rate.",
)
@option(
    "--min-lr",
    default=1e-6,
    show_default=True,
    help="Minimum learning rate before early stopping on training.",
)
@option(
    "--patience", default=10, show_default=True, help="Patience for ReduceLROnPlateau.",
)
@option(
    "--device", default="cpu", show_default=True, help="Device used to train nn",
)
@option(
    "--epochs", default=5, show_default=True, help="Number of epochs for training.",
)
def train(
    model_path,
    data_path,
    loss_out,
    loss_fn,
    batch_size,
    learning_rate,
    min_lr,
    patience,
    device,
    epochs,
):
    """
    Train a model, saving trained model to original location.
    """
    model = torch.load(model_path)
    [_, train_data_list] = from_pickle_file(data_path)

    analysis_params = {
        "model": model,
        "train_data_list": train_data_list,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": device,
    }

    analysis = Analysis(**analysis_params)
    known_loss_fn = {"rmse": rmse, "mse": mse}
    if loss_fn not in known_loss_fn:
        raise IOError(loss_fn + " not known")

    training_params = {
        "epoch_count": epochs,
        "loss_fn": known_loss_fn[loss_fn],
        "patience": patience,
        "min_lr": min_lr,
    }

    click.echo(f"Starting training. {training_params}")
    losses = pd.Series(analysis.train(**training_params))
    torch.save(model, model_path)
    if loss_out is not None:
        losses.to_csv(loss_out)


@cli.command(
    name="eval",
    short_help="Evaluate the performance of a model and dump \
    the a dictionary containing the results",
)
@argument("model_path", type=click.Path(exists=True))
@argument("data_path", type=click.Path(exists=True))
@option("--out", required=True, type=click.Path())
@option("--device", type=str, required=False, default="cpu")
def eval(model_path, data_path, out, device):
    """
    TODO
    """
    click.echo(f"LOG: Loading model from {model_path}")
    model = torch.load(model_path)

    click.echo(f"LOG: loading testing data from {data_path}")
    [test_data, _] = from_pickle_file(data_path)

    click.echo(f"LOG: evaluating test data with given model")
    evaluation = evaluation_dict(model, test_data, device)

    click.echo(f"LOG: pickle dump evalution data dictionary to {out}")
    to_pickle_file(evaluation, out)

    click.echo("eval finished")


@cli.command(
    name="scatter",
    short_help="Evaluate and produce scatter plot of observed vs. predicted \
    targets on the test set provided.",
)
@argument("model_path", type=click.Path(exists=True))
@argument("data_path", type=click.Path(exists=True))
@option("--out", required=True, type=click.Path())
@option("--device", type=str, required=False, default="cpu")
def scatter(model_path, data_path, out, device):
    """
    Evaluate and produce scatter plot of observed vs. predicted
    targets on the test set provided
    """
    click.echo(f"LOG: Loading model from {model_path}")
    model = torch.load(model_path)

    click.echo(f"LOG: loading testing data from {data_path}")
    [test_data, _] = from_pickle_file(data_path)

    click.echo(f"LOG: evaluating test data with given model")
    evaluation = evaluation_dict(model, test_data, device)

    at_least_one = True
    click.echo(f"LOG: plotting scatter correlation")
    plot_test_correlation(evaluation, out)

    click.echo(f"LOG: scatter plot finished and dumped to {out}")


@cli.command(
    name="contour",
    short_help="Evaluate the the latent space of a model with a two \
    dimensional latent space by predicting across grid of values",
)
@argument("model_path", type=click.Path(exists=True))
@option("--start", required=False, type=int, default=0, show_default=True)
@option("--end", required=False, type=int, default=1000, show_default=True)
@option("--nticks", required=False, type=int, default=100, show_default=True)
@option("--out", required=True, type=click.Path())
@option("--device", type=str, required=False, default="cpu")
def contour(model_path, start, end, nticks, out, device):
    """
    Evaluate the the latent space of a model with a two
    dimensional latent space by predicting across grid of values
    """
    click.echo(f"LOG: Loading model from {model_path}")
    model = torch.load(model_path)

    # TODO also check for 2d latent space
    if type(model) != DMSFeedForwardModel:
        raise TypeError("Model must be a DMSFeedForwardModel")

    # TODO add device
    click.echo(f"LOG: plotting contour")
    latent_space_contour_plot_2D(model, out, start, end, nticks)

    click.echo(f"LOG: Contour finished and dumped to {out}")


@cli.command(
    name="beta",
    short_help="This command will plot the beta coeff for each possible mutation \
    at each site along the sequence as a heatmap",
)
@argument("model_path", type=click.Path(exists=True))
@argument("data_path", type=click.Path(exists=True))
@option("--out", required=True, type=click.Path())
def beta(model_path, data_path, out):
    """
    This command will plot the beta coeff for each possible mutation \
    at each site along the sequence as a heatmap
    """
    click.echo(f"LOG: Loading model from {model_path}")
    model = torch.load(model_path)

    # the test data holds some metadata
    click.echo(f"LOG: loading testing data from {data_path}")
    [test_data, _] = from_pickle_file(data_path)
    click.echo(
        f"LOG: loaded data, evaluating beta coeff for wildtype seq: {test_data.wtseq}"
    )

    click.echo(f"LOG: plotting beta coefficients")
    beta_coefficients(model, test_data, out)

    click.echo(f"LOG: Beta coefficients plotted and dumped to {out}")


if __name__ == "__main__":
    cli()
