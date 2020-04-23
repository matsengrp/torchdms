import click
import pandas as pd
import os
import torch

from torchdms.data import *
from torchdms.analysis import Analysis
from torchdms.model import *
from torchdms.loss import *
from torchdms.utils import *
from click import Choice, Path, command, group, option


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
@click.argument("in_path", required=True, type=click.Path(exists=True))
@click.argument("out_path", required=True, type=click.Path())
@click.argument("targets", type=str, nargs=-1, required=True)
@option(
    "--per-stratum-variants-for-test",
    type=int,
    required=False,
    default=100,
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


@cli.command(name="create", short_help="Create a Model")
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path())
@click.argument("model_name")
@click.argument("layers", type=int, nargs=-1, required=False)
def create(model_name, data_path, out_path, layers):
    """
    Create a model. Model name can be the name of any
    of the functions defined in torch.models. 

    If using the BuildYourOwnVanillaNet model, you must provide some number of 
    integer arguments following the model name to specify the number of nodes
    and layers for each (LAYERS argument).
    """
    known_models = {
        "SingleSigmoidNet": SingleSigmoidNet,
        "AdditiveLinearModel": AdditiveLinearModel,
        "TwoByTwoOutputTwoNet": TwoByTwoOutputTwoNet,
        "TwoByTwoNetOutputOne": TwoByTwoNetOutputOne,
        "BuildYourOwnVanillaNet": BuildYourOwnVanillaNet,
    }
    click.echo(f"LOG: searching for {model_name}")
    if model_name not in known_models:
        raise IOError(model_name + " not known")
    click.echo(f"LOG: found {model_name}")
    click.echo(f"LOG: loading training data")
    [test_BMD, _] = from_pickle_file(data_path)

    click.echo(f"LOG: Test data input size: {test_BMD.feature_count()}")
    click.echo(f"LOG: Test data output size: {test_BMD.targets.shape[1]}")
    if model_name == "BuildYourOwnVanillaNet":
        if len(list(layers)) == 0:
            raise ValueError("Must provide layers to define custom model")
        for layer in layers:
            if type(layer) != int:
                raise TypeError("All layer input must be integers")
        model = BuildYourOwnVanillaNet(
            test_BMD.feature_count(), list(layers), test_BMD.targets.shape[1]
        )
        click.echo(f"LOG: Successfully made custom model")
        torch.save(model, out_path)
    else:
        model = known_models[model_name](test_BMD.feature_count())
        click.echo(f"LOG: Successfully made {model_name} model")
        torch.save(model, out_path)
    click.echo(f"LOG: Model defined as: {model}")
    click.echo(f"LOG: Saved model to {out_path}")


# TODO train should just take the config file as a dict
@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--loss-out", type=click.Path(), required=False)
@click.option(
    "--loss-fn", default="rmse", required=True, help="Loss function for training."
)
@click.option(
    "--batch-size", default=500, show_default=True, help="Batch size for training.",
)
@click.option(
    "--learning-rate", default=1e-3, show_default=True, help="Initial learning rate.",
)
@click.option(
    "--min-lr",
    default=1e-6,
    show_default=True,
    help="Minimum learning rate before early stopping on training.",
)
@click.option(
    "--patience", default=10, show_default=True, help="Patience for ReduceLROnPlateau.",
)
@click.option(
    "--device", default="cpu", show_default=True, help="Device used to train nn",
)
@click.option(
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


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@option("--eval-dict-out", required=False, type=click.Path())
@option("--scatter-plot-out", required=False, type=click.Path())
@option("--device", type=str, required=False, default="cpu")
def eval(model_path, data_path, eval_dict_out, scatter_plot_out, device):
    """
    Evaluate a model given some testing data in the form of a BinaryMapDataset
    object. You can output any of the options below.
    """
    click.echo(f"LOG: Loading model from {model_path}")
    model = torch.load(model_path)

    click.echo(f"LOG: loading testing data from {data_path}")
    [test_data, _] = from_pickle_file(data_path)

    click.echo(f"LOG: evaluating test data with given model")
    evaluation = evaluatation_dict(model, test_data, device)

    at_least_one = False
    if eval_dict_out is not None:
        at_least_one = True
        click.echo(f"LOG: pickle dump evalution data dictionary to {eval_dict_out}")
        to_pickle_file(evaluation, eval_dict_out)

    if scatter_plot_out is not None:
        at_least_one = True
        click.echo(f"LOG: plotting scatter correlation")
        plot_test_correlation(evaluation, scatter_plot_out)
        pass

    if not at_least_one:
        click.echo(
            f"WARNING: No outputs specified. You essentially did nothing \
        ... ¯\_(ツ)_/¯"
        )
    click.echo("eval finished")


if __name__ == "__main__":
    cli()
