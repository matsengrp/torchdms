"""The command line interface."""
import pathlib
import json
import os
import click
import click_config_file
import torch
from torchdms.analysis import Analysis
from torchdms.data import (
    partition,
    prep_by_stratum_and_export,
)
from torchdms.evaluation import (
    build_evaluation_dict,
    complete_error_summary,
    error_df_of_evaluation_dict,
)
from torchdms.model import (
    model_of_string,
    VanillaGGE,
)
from torchdms.loss import l1, mse, rmse
from torchdms.utils import (
    from_pickle_file,
    from_json_file,
    make_cartesian_product_hierarchy,
    to_pickle_file,
)
from torchdms.plot import (
    beta_coefficients,
    latent_space_contour_plot_2d,
    plot_error,
    plot_test_correlation,
)


def json_provider(file_path, cmd_name):
    """Enable loading of flags from a JSON file via click_config_file."""
    if cmd_name:
        with open(file_path) as config_data:
            config_dict = json.load(config_data)
            if cmd_name not in config_dict:
                if "default" in config_dict:
                    return config_dict["default"]
                # else:
                raise IOError(
                    f"Could not find a '{cmd_name}' or 'default' section in '{file_path}'"
                )
            return config_dict[cmd_name]
    # else:
    return None


def process_dry_run(ctx, method_name, local_variables):
    """Return whether ctx tells us we are doing a dry run; print the call."""
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
    """Train and evaluate neural networks on deep mutational scanning data."""
    ctx.ensure_object(dict)
    ctx.obj["dry_run"] = dry_run


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
    help="This is the number of variants for each stratum to hold out for testing, with "
    "the same number used for validation. The rest of the examples will be used for "
    "training the model.",
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
@click.option(
    "--export-dataframe",
    type=str,
    required=False,
    default=None,
    help="Filename prefix for exporting the original dataframe in a .pkl file with an "
    "appended in_test column.",
)
@click.option(
    "--partition-by",
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
    partition_by,
):
    """Prepare data for training.

    IN_PATH should point to a pickle dump'd Pandas DataFrame containing
    the string encoded `aa_substitutions` column along with any TARGETS
    you specify. OUT_PREFIX is the location to dump the prepped data to
    another pickle file.
    """
    if process_dry_run(ctx, "prep", locals()):
        return
    click.echo(f"LOG: Targets: {targets}")
    click.echo(f"LOG: Loading substitution data for: {in_path}")
    aa_func_scores, wtseq = from_pickle_file(in_path)
    click.echo("LOG: Successfully loaded data")

    total_variants = len(aa_func_scores.iloc[:, 1])
    click.echo(f"LOG: There are {total_variants} total variants in this dataset")

    if partition_by is None and "library" in aa_func_scores.columns:
        click.echo(
            "WARNING: you have a 'library' column but haven't specified a partition "
            "via '--partition-by'"
        )

    def prep_by_stratum_and_export_of_partition_label_and_df(partition_label, df):
        split_df = partition(
            df,
            per_stratum_variants_for_test,
            skip_stratum_if_count_is_smaller_than,
            export_dataframe,
            partition_label,
        )

        prep_by_stratum_and_export(
            split_df, wtseq, targets, out_prefix, str(ctx.params), partition_label,
        )

    if partition_by in aa_func_scores.columns:
        for partition_label, per_partition_label_df in aa_func_scores.groupby(
            partition_by
        ):
            click.echo(f"LOG: Partitioning data via '{partition_label}'")
            prep_by_stratum_and_export_of_partition_label_and_df(
                partition_label, per_partition_label_df.copy()
            )
    else:
        prep_by_stratum_and_export_of_partition_label_and_df(None, aa_func_scores)

    click.echo(
        "LOG: Successfully finished prep and dumped BinaryMapDataset "
        f"object to {out_prefix}"
    )


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path())
@click.argument("model_string")
@click.option(
    "--monotonic",
    type=float,
    default=None,
    help="If this option is used, "
    "then the model will be initialized with weights greater than zero. "
    "During training with this model then, tdms will put a floor of "
    "0 on all non-bias weights. It will also multiply the output by the value provided "
    " as an option argument here, so use -1.0 if you want your nonlinearity to be "
    "monotonically decreasing, or 1.0 if you want it to be increasing.",
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
    """Create a model.

    Model string describes the model, such as 'VanillaGGE(1,10)'.
    """
    if process_dry_run(ctx, "create", locals()):
        return

    model = model_of_string(model_string, data_path, monotonic)
    model.beta_l1_coefficient = beta_l1_coefficient

    torch.save(model, out_path)
    click.echo(f"LOG: Model defined as: {model}")
    click.echo(f"LOG: Saved model to {out_path}")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option(
    "--loss-fn", default="l1", show_default=True, help="Loss function for training."
)
@click.option(
    "--loss-weight-span",
    type=float,
    default=None,
    help="If this option is used, add a weight to a mean-absolute-deviation loss equal "
    "to the exponential of a loss decay times the true score.",
)
@click.option(
    "--batch-size", default=500, show_default=True, help="Batch size for training.",
)
@click.option(
    "--learning-rate", default=1e-3, show_default=True, help="Initial learning rate.",
)
@click.option(
    "--min-lr",
    default=1e-5,
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
    "--independent-starts",
    default=5,
    show_default=True,
    help="Number of independent training starts to use. Each training start gets trained "
    "10% of the full number of epochs and the best start is used for full training.",
)
@click.option(
    "--epochs",
    default=5,
    show_default=True,
    help="Number of epochs for full training.",
)
@click_config_file.configuration_option(implicit=False, provider=json_provider)
@click.pass_context
def train(
    ctx,
    model_path,
    data_path,
    loss_fn,
    loss_weight_span,
    batch_size,
    learning_rate,
    min_lr,
    patience,
    device,
    independent_starts,
    epochs,
):
    """Train a model, saving trained model to original location."""
    if process_dry_run(ctx, "train", locals()):
        return
    model = torch.load(model_path)
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
        raise IOError(loss_fn + " not known")

    if loss_weight_span is not None:
        click.echo(
            "NOTE: you are using loss decays, which assumes that you want to up-weight "
            "the loss function when the true target value is large. Is that true?"
        )

    training_params = {
        "independent_start_count": independent_starts,
        "epoch_count": epochs,
        "loss_fn": known_loss_fn[loss_fn],
        "patience": patience,
        "min_lr": min_lr,
        "loss_weight_span": loss_weight_span,
    }

    click.echo(f"Starting training. {training_params}")
    analysis.multi_train(**training_params)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--device", type=str, required=False, default="cpu")
@click_config_file.configuration_option(implicit=False, provider=json_provider)
@click.pass_context
def evaluate(ctx, model_path, data_path, out, device):
    """Evaluate the performance of a model.

    Dump to a dictionary containing the results.
    """
    if process_dry_run(ctx, "evaluate", locals()):
        return
    click.echo(f"LOG: Loading model from {model_path}")
    model = torch.load(model_path)

    click.echo(f"LOG: loading testing data from {data_path}")
    data = from_pickle_file(data_path)

    click.echo("LOG: evaluating test data with given model")
    evaluation = build_evaluation_dict(model, data.test, device)

    click.echo(f"LOG: pickle dump evalution data dictionary to {out}")
    to_pickle_file(evaluation, out)

    click.echo("evaluate finished")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option(
    "--show-points", is_flag=True, help="Show points in addition to LOWESS curves.",
)
@click.option("--device", type=str, required=False, default="cpu")
@click_config_file.configuration_option(implicit=False, provider=json_provider)
@click.pass_context
def error(ctx, model_path, data_path, out, show_points, device):
    """Evaluate and produce plot of error."""
    if process_dry_run(ctx, "error", locals()):
        return
    click.echo(f"LOG: Loading model from {model_path}")
    model = torch.load(model_path)
    prefix = os.path.splitext(out)[0]

    click.echo(f"LOG: loading testing data from {data_path}")
    data = from_pickle_file(data_path)

    evaluation = build_evaluation_dict(model, data.test, device)
    error_df = error_df_of_evaluation_dict(evaluation)

    plot_error(error_df, out, show_points)
    error_df.to_csv(prefix + ".csv", index=False)

    error_summary_df = complete_error_summary(data, model)
    error_summary_df.to_csv(prefix + "-summary.csv")

    click.echo(f"LOG: error plot finished and dumped to {out}")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--device", type=str, required=False, default="cpu")
@click_config_file.configuration_option(implicit=False, provider=json_provider)
@click.pass_context
def scatter(ctx, model_path, data_path, out, device):
    """Evaluate and produce scatter plot of observed vs predicted targets on
    the test set provided."""
    if process_dry_run(ctx, "scatter", locals()):
        return
    click.echo(f"LOG: Loading model from {model_path}")
    model = torch.load(model_path)

    click.echo(f"LOG: loading testing data from {data_path}")
    data = from_pickle_file(data_path)

    click.echo("LOG: evaluating test data with given model")
    evaluation = build_evaluation_dict(model, data.test, device)

    click.echo("LOG: plotting scatter correlation")
    plot_test_correlation(evaluation, model, out)

    click.echo(f"LOG: scatter plot finished and dumped to {out}")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--start", required=False, type=int, default=0, show_default=True)
@click.option("--end", required=False, type=int, default=1000, show_default=True)
@click.option("--nticks", required=False, type=int, default=100, show_default=True)
@click.option("--out", required=True, type=click.Path())
@click_config_file.configuration_option(implicit=False, provider=json_provider)
@click.pass_context
def contour(ctx, model_path, start, end, nticks, out):
    """Visualize the the latent space of a model.

    Make a contour plot with a two dimensional latent space by
    predicting across grid of values.
    """
    if process_dry_run(ctx, "contour", locals()):
        return
    click.echo(f"LOG: Loading model from {model_path}")
    model = torch.load(model_path)

    if not isinstance(model, VanillaGGE):
        raise TypeError("Model must be a VanillaGGE")

    click.echo("LOG: plotting contour")
    latent_space_contour_plot_2d(model, out, start, end, nticks)

    click.echo(f"LOG: Contour finished and dumped to {out}")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click_config_file.configuration_option(implicit=False, provider=json_provider)
@click.pass_context
def beta(ctx, model_path, data_path, out):
    """Plot beta coefficients as a heatmap."""
    if process_dry_run(ctx, "beta", locals()):
        return
    click.echo(f"LOG: Loading model from {model_path}")
    model = torch.load(model_path)

    # the test data holds some metadata
    click.echo(f"LOG: loading testing data from {data_path}")
    data = from_pickle_file(data_path)
    click.echo(
        f"LOG: loaded data, evaluating beta coeff for wildtype seq: {data.test.wtseq}"
    )

    click.echo("LOG: plotting beta coefficients")
    beta_coefficients(model, data.test, out)

    click.echo(f"LOG: Beta coefficients plotted and dumped to {out}")


def restrict_dict_to_params(d_to_restrict, cmd):
    """Restrict the given dictionary to the names of parameters for cmd."""
    param_names = {param.name for param in cmd.params}
    return {key: d_to_restrict[key] for key in d_to_restrict if key in param_names}


@cli.command()
@click_config_file.configuration_option(
    implicit=False, required=True, provider=json_provider
)
@click.pass_context
def go(ctx):
    """Run a common sequence of commands: create, train, scatter, and beta.

    Then touch a `.sentinel` file to signal successful completion.
    """
    prefix = ctx.default_map["prefix"]
    model_path = prefix + ".model"
    ctx.invoke(
        create, out_path=model_path, **restrict_dict_to_params(ctx.default_map, create),
    )
    ctx.invoke(
        train, model_path=model_path, **restrict_dict_to_params(ctx.default_map, train),
    )
    error_path = prefix + ".error.pdf"
    ctx.invoke(
        error,
        model_path=model_path,
        out=error_path,
        **restrict_dict_to_params(ctx.default_map, error),
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
    sentinel_path = prefix + ".sentinel"
    click.echo(f"LOG: `tdms go` completed; touching {sentinel_path}")
    pathlib.Path(sentinel_path).touch()


@cli.command()
@click.argument("choice_json_path", required=True, type=click.Path(exists=True))
@click.pass_context
def cartesian(ctx, choice_json_path):
    """Take the cartesian product of the variable options in a config file."""
    make_cartesian_product_hierarchy(
        from_json_file(choice_json_path), ctx.obj["dry_run"]
    )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
