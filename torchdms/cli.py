import inspect
import os
import re
import click
import json
import pandas as pd
import torch


from click import group, option, argument
import click_config_file
from torchdms.analysis import Analysis
from torchdms.data import prepare, partition
from torchdms.model import VanillaGGE
from torchdms.loss import rmse, mse
from torchdms.utils import (
    evaluation_dict,
    from_pickle_file,
    from_json_file,
    make_cartesian_product_hierarchy,
    to_pickle_file,
    monotonic_params_from_latent_space,
)
from torchdms.plot import (
    beta_coefficients,
    latent_space_contour_plot_2D,
    plot_test_correlation,
)


def json_provider(file_path, cmd_name):
    """
    Enable loading of flags from a JSON file via click_config_file.
    """
    if cmd_name:
        with open(file_path) as config_data:
            config_dict = json.load(config_data)
            if cmd_name not in config_dict:
                raise IOError(f"Could not find a '{cmd_name}' section in '{file_path}'")
            return config_dict[cmd_name]
    # else:
    return None


def process_dry_run(ctx, method_name, local_variables):
    """
    Return whether ctx tells us we are doing a dry run; print the call.
    """
    if "ctx" in local_variables:
        del local_variables["ctx"]
    if ctx.obj["dry_run"]:
        print(f"{method_name}{local_variables})")
        return True
    # else:
    return False


# Entry point
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--dry-run",
    is_flag=True,
    help="Only print paths and files to be made, rather than actually making them.",
)
@click.pass_context
def cli(ctx, dry_run):
    """
    Train and evaluate neural networks on deep mutational scanning data.
    """
    ctx.ensure_object(dict)
    ctx.obj["dry_run"] = dry_run
    pass


@cli.command()
@click.argument("in_path", required=True, type=click.Path(exists=True))
@click.argument("out_prefix", required=True, type=click.Path())
@click.argument("targets", type=str, nargs=-1, required=True)
@click.option(
    "--per-stratum-variants-for-test",
    type=int,
    required=False,
    default=100,
    show_default=True,
    help="This is the number of variants for each stratum to hold out for testing. The "
    "rest of the examples will be used for training the model.",
)
@click.option(
    "--skip-stratum-if-count-is-smaller-than",
    type=int,
    required=False,
    default=250,
    show_default=True,
    help="If the total number of examples for any particular stratum is lower than this "
    "number, we throw out the stratum completely.",
)
@option(
    "--export-dataframe",
    type=str,
    required=False,
    default=None,
    help="Filename prefix for exporting the original dataframe in a .pkl file with an "
    "appended in_test column.",
)
@option(
    "--split-by",
    type=str,
    required=False,
    default=None,
    help="Column name containing a feature by which the data should be split into "
    "independent datasets for partitioning; e.g. 'library'.",
)
@click_config_file.configuration_option(implicit=False, provider=json_provider)
@click.pass_context
def prep(
    ctx,
    in_path,
    out_prefix,
    targets,
    per_stratum_variants_for_test,
    skip_stratum_if_count_is_smaller_than,
    export_dataframe,
    split_by,
):
    """
    Prepare data for training.

    IN_PATH should point to a pickle dump'd Pandas DataFrame containing the string
    encoded `aa_substitutions` column along with any TARGETS you specify.
    OUT_PREFIX is the location to dump the prepped data to another pickle file.
    """
    if process_dry_run(ctx, "prep", locals()):
        return
    click.echo(f"LOG: Targets: {targets}")
    click.echo(f"LOG: Loading substitution data for: {in_path}")
    aa_func_scores, wtseq = from_pickle_file(in_path)
    click.echo(f"LOG: Successfully loaded data")

    total_variants = len(aa_func_scores.iloc[:, 1])
    click.echo(f"LOG: There are {total_variants} total variants in this dataset")

    if split_by in aa_func_scores.columns:
        # feature = name of library or general feature
        # grouped = the subsetted df
        for feature, grouped in aa_func_scores.groupby(split_by):
            click.echo(f"LOG: Partitioning data via '{feature}'")
            test_partition, partitioned_train_data = partition(
                grouped,
                per_stratum_variants_for_test,
                skip_stratum_if_count_is_smaller_than,
                export_dataframe,
                feature,
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

            feature_filename = feature.replace(" ", "_")
            feature_filename = "".join(x for x in feature_filename if x.isalnum())

            to_pickle_file(
                prepare(test_partition, partitioned_train_data, wtseq, list(targets)),
                f"{out_prefix}_{feature_filename}.pkl",
            )

    else:
        test_partition, partitioned_train_data = partition(
            aa_func_scores,
            per_stratum_variants_for_test,
            skip_stratum_if_count_is_smaller_than,
            export_dataframe,
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
            prepare(test_partition, partitioned_train_data, wtseq, list(targets)),
            f"{out_prefix}.pkl",
        )

    click.echo(
        f"LOG: Successfully finished prep and dumped BinaryMapDataset \
          object to {out_prefix}"
    )


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path())
@click.argument("model_string")
@click.option(
    "--monotonic",
    is_flag=True,
    help="If this flag is used, "
    "then the model will be initialized with weights greater than zero. "
    "During training with this model then, tdms will put a floor of "
    "0 on all non-bias weights.",
)
@click.option(
    "--beta-l1-coefficient",
    type=float,
    default=0.0,
    show_default=True,
    help="Coefficient with which to l1-regularize all beta coefficients except for "
    "those to the first latent dimension.",
)
@click_config_file.configuration_option(implicit=False, provider=json_provider)
@click.pass_context
def create(ctx, model_string, data_path, out_path, monotonic, beta_l1_coefficient):
    """
    Create a model.

    Model string describes the model, such as 'VanillaGGE(1,10)'.
    """
    if process_dry_run(ctx, "create", locals()):
        return
    known_models = {
        "VanillaGGE": VanillaGGE,
    }
    try:
        model_regex = re.compile(r"(.*)\((.*)\)")
        match = model_regex.match(model_string)
        model_name = match.group(1)
        layers = list(map(int, match.group(2).split(",")))
    except Exception:
        click.echo(f"ERROR: Couldn't parse model description: '{model_string}'")
        raise
    click.echo(f"LOG: searching for {model_name}")
    if model_name not in known_models:
        raise IOError(model_name + " not known")
    click.echo(f"LOG: found {model_name}")
    click.echo(f"LOG: loading training data")
    [test_BMD, _] = from_pickle_file(data_path)

    click.echo(f"LOG: Test data input size: {test_BMD.feature_count()}")
    click.echo(f"LOG: Test data output size: {test_BMD.targets.shape[1]}")
    if model_name == "VanillaGGE":
        if len(layers) == 0:
            click.echo(f"LOG: No layers provided means creating a linear model")
        for layer in layers:
            if not isinstance(layer, int):
                raise TypeError("All layer input must be integers")
        model = VanillaGGE(
            test_BMD.feature_count(),
            layers,
            test_BMD.targets.shape[1],
            beta_l1_coefficient=beta_l1_coefficient,
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


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--loss-out", type=click.Path(), required=False)
@click.option(
    "--loss-fn", default="rmse", show_default=True, help="Loss function for training."
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
@click_config_file.configuration_option(implicit=False, provider=json_provider)
@click.pass_context
def train(
    ctx,
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
    if process_dry_run(ctx, "train", locals()):
        return
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
@click.option("--out", required=True, type=click.Path())
@click.option("--device", type=str, required=False, default="cpu")
@click_config_file.configuration_option(implicit=False, provider=json_provider)
@click.pass_context
def eval(ctx, model_path, data_path, out, device):
    """
    Evaluate the performance of a model.

    Dump to a dictionary containing the results.
    """
    if process_dry_run(ctx, "eval", locals()):
        return
    click.echo(f"LOG: Loading model from {model_path}")
    model = torch.load(model_path)

    click.echo(f"LOG: loading testing data from {data_path}")
    [test_data, _] = from_pickle_file(data_path)

    click.echo(f"LOG: evaluating test data with given model")
    evaluation = evaluation_dict(model, test_data, device)

    click.echo(f"LOG: pickle dump evalution data dictionary to {out}")
    to_pickle_file(evaluation, out)

    click.echo("eval finished")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--device", type=str, required=False, default="cpu")
@click_config_file.configuration_option(implicit=False, provider=json_provider)
@click.pass_context
def scatter(ctx, model_path, data_path, out, device):
    """
    Evaluate and produce scatter plot of observed vs. predicted targets on the test set
    provided.
    """
    if process_dry_run(ctx, "scatter", locals()):
        return
    click.echo(f"LOG: Loading model from {model_path}")
    model = torch.load(model_path)

    click.echo(f"LOG: loading testing data from {data_path}")
    [test_data, _] = from_pickle_file(data_path)

    click.echo(f"LOG: evaluating test data with given model")
    evaluation = evaluation_dict(model, test_data, device)

    click.echo(f"LOG: plotting scatter correlation")
    plot_test_correlation(evaluation, model, out)

    click.echo(f"LOG: scatter plot finished and dumped to {out}")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--start", required=False, type=int, default=0, show_default=True)
@click.option("--end", required=False, type=int, default=1000, show_default=True)
@click.option("--nticks", required=False, type=int, default=100, show_default=True)
@click.option("--out", required=True, type=click.Path())
@click.option("--device", type=str, required=False, default="cpu")
@click_config_file.configuration_option(implicit=False, provider=json_provider)
@click.pass_context
def contour(ctx, model_path, start, end, nticks, out, device):
    """
    Visualize the the latent space of a model.

    Make a contour plot with a two dimensional latent space by predicting across grid of
    values.
    """
    if process_dry_run(ctx, "contour", locals()):
        return
    click.echo(f"LOG: Loading model from {model_path}")
    model = torch.load(model_path)

    # TODO also check for 2d latent space
    if not isinstance(model, VanillaGGE):
        raise TypeError("Model must be a VanillaGGE")

    # TODO add device
    click.echo(f"LOG: plotting contour")
    latent_space_contour_plot_2D(model, out, start, end, nticks)

    click.echo(f"LOG: Contour finished and dumped to {out}")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click_config_file.configuration_option(implicit=False, provider=json_provider)
@click.pass_context
def beta(ctx, model_path, data_path, out):
    """
    Plot beta coefficients as a heatmap.
    """
    if process_dry_run(ctx, "beta", locals()):
        return
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


def restrict_dict_to_params(d, cmd):
    """
    Restrict the given dictionary to the names of parameters for cmd.
    """
    param_names = {param.name for param in cmd.params}
    return {key: d[key] for key in d if key in param_names}


@cli.command()
@click_config_file.configuration_option(
    implicit=False, required=True, provider=json_provider
)
@click.pass_context
def go(ctx):
    """
    Run a common sequence of commands: create, train, scatter, and beta.
    """
    prefix = ctx.default_map["prefix"]
    model_path = prefix + ".model"
    ctx.invoke(
        create, out_path=model_path, **restrict_dict_to_params(ctx.default_map, create),
    )
    loss_path = prefix + ".loss.csv"
    ctx.invoke(
        train,
        model_path=model_path,
        loss_out=loss_path,
        **restrict_dict_to_params(ctx.default_map, train),
    )
    scatter_path = prefix + ".scatter.pdf"
    ctx.invoke(
        scatter,
        model_path=model_path,
        out=scatter_path,
        **restrict_dict_to_params(ctx.default_map, scatter),
    )
    beta_path = prefix + ".beta.pdf"
    ctx.invoke(
        beta,
        model_path=model_path,
        out=beta_path,
        **restrict_dict_to_params(ctx.default_map, beta),
    )


@cli.command()
@click.argument("choice_json_path", required=True, type=click.Path(exists=True))
@click.pass_context
def cartesian(ctx, choice_json_path):
    """
    Take the cartesian product of the variable options in a config file.
    """
    make_cartesian_product_hierarchy(
        from_json_file(choice_json_path), ctx.obj["dry_run"]
    )


if __name__ == "__main__":
    cli()
