"""Command line interface."""

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
    from_pickle_file,
    from_json_file,
    make_cartesian_product_hierarchy,
    to_pickle_file,
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


def set_random_seed(seed):
    if seed is not None:
        click.echo(f"LOG: Setting random seed to {seed}.")
        torch.manual_seed(seed)
        random.seed(seed)


def dry_run_option(command):
    return click.option(
        "--dry-run",
        is_flag=True,
        help="Only print paths and files to be made, rather than actually making them.",
    )(command)


def print_method_name_and_locals(method_name, local_variables):
    """Print method name and local variables."""
    if "ctx" in local_variables:
        del local_variables["ctx"]
    print(f"{method_name}{local_variables})")


def seed_option(command):
    return click.option(
        "--seed",
        type=int,
        show_default=True,
        help="Set random seed. Seed is uninitialized if not set.",
    )(command)


# Entry point
@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.option(
    "-v",
    "--version",
    is_flag=True,
    help="Print version and exit. Note that as per `git describe`, the SHA is prefixed "
    "by a `g`.",
)
def cli(version):
    """Train and evaluate neural networks on deep mutational scanning data."""
    if version:
        print(f"torchdms version {torchdms.__version__}")


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
    "--drop-nans",
    is_flag=True,
    help="Drop all rows that contain a nan.",
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
@dry_run_option
@seed_option
@click_config_file.configuration_option(implicit=False, provider=json_provider)
@click.pass_context
def prep(
    ctx,
    in_path,
    out_prefix,
    targets,
    per_stratum_variants_for_test,
    skip_stratum_if_count_is_smaller_than,
    drop_nans,
    export_dataframe,
    partition_by,
    dry_run,
    seed,
):
    """Prepare data for training.

    IN_PATH should point to a pickle dump'd Pandas DataFrame containing
    the string encoded `aa_substitutions` column along with any TARGETS
    you specify. OUT_PREFIX is the location to dump the prepped data to
    another pickle file.
    """
    if dry_run:
        print_method_name_and_locals("prep", locals())
        return
    set_random_seed(seed)
    click.echo(f"LOG: Targets: {targets}")
    aa_func_scores, wtseq = from_pickle_file(in_path)
    if drop_nans:
        click.echo("LOG: dropping NaNs as requested.")
        aa_func_scores.dropna(inplace=True)
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
            split_df,
            wtseq,
            targets,
            out_prefix,
            str(ctx.params),
            partition_label,
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
        "LOG: Successfully finished prep and dumped SplitDataset "
        f"object to {out_prefix}"
    )


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option(
    "--out-prefix",
    type=click.Path(),
    help="If this flag is set, make pdf plots summarizing the data.",
)
@click_config_file.configuration_option(implicit=False, provider=json_provider)
def summarize(data_path, out_prefix):
    """Report various summaries of the data."""
    data = from_pickle_file(data_path)
    if isinstance(data, list):
        summarize_dms_variants_dataframe(data[0], out_prefix)
    if isinstance(data, pd.DataFrame):
        summarize_dms_variants_dataframe(data, out_prefix)
    elif isinstance(data, SplitDataset):
        data.summarize(out_prefix)
    else:
        raise NotImplementedError(f"Summary of {type(data).__name__}")


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
def validate(data_path):
    """Validate that a given data set is sane."""
    splitdata = from_pickle_file(data_path)
    for _, data in splitdata.labeled_splits:
        check_onehot_encoding(data)
    click.echo(f"Validated {data_path}")


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
    "--beta-l1-coefficients",
    type=str,
    help="Coefficients with which to l1-regularize beta coefficients, "
    "a comma-seperated list of coefficients for each latent dimension.",
)
@click.option(
    "--interaction-l1-coefficients",
    type=str,
    help="Coefficients with which to l1-regularize site interaction weights, "
    "a comma-seperated list of coefficients for each latent dimension",
)
@seed_option
@click_config_file.configuration_option(implicit=False, provider=json_provider)
def create(
    model_string,
    data_path,
    out_path,
    monotonic,
    beta_l1_coefficients,
    interaction_l1_coefficients,
    seed,
):
    """Create a model.

    Model string describes the model, such as 'Planifolia(1,10)'.
    """
    set_random_seed(seed)
    beta_l1_coefficients = [float(x) for x in beta_l1_coefficients.split(",")]
    kwargs = dict(monotonic_sign=monotonic)
    if len(beta_l1_coefficients) == 1:
        kwargs["beta_l1_coefficient"] = beta_l1_coefficients[0]
    else:
        kwargs["beta_l1_coefficients"] = beta_l1_coefficients
    interaction_l1_coefficients = [
        float(x) for x in interaction_l1_coefficients.split(",")
    ]
    if len(interaction_l1_coefficients) == 1:
        kwargs["interaction_l1_coefficient"] = interaction_l1_coefficients[0]
    else:
        kwargs["interaction_l1_coefficients"] = interaction_l1_coefficients

    model = model_of_string(
        model_string,
        data_path,
        **kwargs,
    )

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
    "--batch-size",
    default=500,
    show_default=True,
    help="Batch size for training.",
)
@click.option(
    "--learning-rate",
    default=1e-3,
    show_default=True,
    help="Initial learning rate.",
)
@click.option(
    "--min-lr",
    default=1e-5,
    show_default=True,
    help="Minimum learning rate before early stopping on training.",
)
@click.option(
    "--patience",
    default=10,
    show_default=True,
    help="Patience for ReduceLROnPlateau.",
)
@click.option(
    "--device",
    default="cpu",
    show_default=True,
    help="Device used to train nn",
)
@click.option(
    "--independent-starts",
    default=5,
    show_default=True,
    help="Number of independent training starts to use. Each training start gets trained "
    "independently and the best start is used for full training.",
)
@click.option(
    "--independent-start-epochs",
    type=int,
    help="How long to train each independent start. If not set, 10% of the full number "
    "of epochs is used.",
)
@click.option(
    "--simple-training",
    is_flag=True,
    help="Ignore all fancy training options: do bare-bones training for a fixed number "
    "of epochs. Fail if data contains nans.",
)
@click.option(
    "--exp-target",
    type=float,
    default=None,
    help="Provide base to be exponentiated by functional scores of variants."
    "Emphasizes fitting highly functional variants. If on, weight decay will be turned off.",
)
@click.option(
    "--beta-rank",
    type=int,
    default=None,
    help="What number of dimensions to use in the low-rank reconstructions of betas.",
)
@click.option(
    "--epochs",
    default=100,
    show_default=True,
    help="Number of epochs for full training.",
)
@dry_run_option
@seed_option
@click_config_file.configuration_option(implicit=False, provider=json_provider)
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


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--device", type=str, required=False, default="cpu")
@click_config_file.configuration_option(implicit=False, provider=json_provider)
def evaluate(model_path, data_path, out, device):
    """Evaluate the performance of a model.

    Dump to a dictionary containing the results.
    """
    model = torch.load(model_path)
    data = from_pickle_file(data_path)
    evaluation = build_evaluation_dict(model, data.test, device)
    click.echo(f"LOG: pickle dump evalution data dictionary to {out}")
    to_pickle_file(evaluation, out)


def default_map_of_ctx_or_parent(ctx):
    """Get the default_map from this context or the parent context.

    In our application, the default_map is parsed from the JSON
    configuration file. The parent context can be useful if we are
    invoked from another subcommand.
    """
    default_map = ctx.default_map
    if default_map is None:
        default_map = ctx.parent.default_map
    return default_map


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option(
    "--show-points",
    is_flag=True,
    help="Show points in addition to LOWESS curves.",
)
@click.option("--device", type=str, required=False, default="cpu")
@click.option(
    "--include-details",
    is_flag=True,
    help="Include details from config file in error summary.",
)
@click_config_file.configuration_option(implicit=False, provider=json_provider)
@click.pass_context
def error(ctx, model_path, data_path, out, show_points, device, include_details):
    """Evaluate and produce plot of error."""
    model = torch.load(model_path)
    prefix = os.path.splitext(out)[0]

    data = from_pickle_file(data_path)

    evaluation = build_evaluation_dict(model, data.test, device)
    error_df = error_df_of_evaluation_dict(evaluation)

    plot_error(error_df, out, model.str_summary(), show_points)
    error_df.to_csv(prefix + ".csv", index=False)

    error_summary_df = complete_error_summary(data, model)
    if include_details:
        default_map = default_map_of_ctx_or_parent(ctx)
        if default_map is not None:
            for key, value in default_map.items():
                error_summary_df[key] = value
    error_summary_df.to_csv(prefix + "-summary.csv")

    click.echo(f"LOG: error plot finished and dumped to {out}")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--device", type=str, required=False, default="cpu")
@click_config_file.configuration_option(implicit=False, provider=json_provider)
def scatter(model_path, data_path, out, device):
    """Evaluate and produce scatter plot of observed vs predicted targets on
    the test set provided."""
    model = torch.load(model_path)
    data = from_pickle_file(data_path)
    evaluation = build_evaluation_dict(model, data.test, device)
    plot_test_correlation(evaluation, model, out)
    click.echo(f"LOG: scatter plot finished and dumped to {out}")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click_config_file.configuration_option(implicit=False, provider=json_provider)
def beta(model_path, data_path, out):
    """Plot beta coefficients as a heatmap."""
    model = torch.load(model_path)
    data = from_pickle_file(data_path)
    click.echo(
        f"LOG: loaded data, evaluating beta coeff for wildtype seq: {data.test.wtseq}"
    )
    beta_coefficients(model, data.test, out)
    click.echo(f"LOG: Beta coefficients plotted and dumped to {out}")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click_config_file.configuration_option(implicit=False, provider=json_provider)
def heatmap(model_path, out):
    """Plot single mutant predictions as a heatmap."""
    model = torch.load(model_path)
    plot_heatmap(model, out)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--steps", required=False, type=int, default=100, show_default=True)
@click.option("--out", required=True, type=click.Path())
@click.option("--device", type=str, required=False, default="cpu")
@click_config_file.configuration_option(implicit=False, provider=json_provider)
def geplot(model_path, data_path, steps, out, device):
    """Make a "global epistasis" plot showing the fit to the nonlinearity."""
    model = torch.load(model_path)
    data = from_pickle_file(data_path)
    geplot_df = build_geplot_df(model, data.test, device)
    if model.latent_dim == 1:
        plot_geplot(geplot_df, out, model.str_summary())
    elif model.latent_dim == 2:
        nonlinearity_df = build_2d_nonlinearity_df(model, geplot_df, steps)
        plot_2d_geplot(model, geplot_df, nonlinearity_df, out)
    else:
        click.echo(
            "WARNING: I don't know how to make a GE plot for any other dims other than 1 "
            "and 2."
        )


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click_config_file.configuration_option(implicit=False, provider=json_provider)
def svd(model_path, data_path, out):
    """Plot singular values of beta matricies."""
    model = torch.load(model_path)
    data = from_pickle_file(data_path)
    click.echo("LOG: model loaded, calculating SVD for beta coefficents.")
    plot_svd(model, data.test, out)
    click.echo(f"LOG: Singular values of beta plotted and dumped to {out}")


def restrict_dict_to_params(d_to_restrict, cmd):
    """Restrict the given dictionary to the names of parameters for cmd."""
    param_names = {param.name for param in cmd.params}
    return {key: d_to_restrict[key] for key in d_to_restrict if key in param_names}


@cli.command()
@click_config_file.configuration_option(implicit=False, provider=json_provider)
@click.pass_context
def go(ctx):
    """Run a common sequence of commands: create, train, scatter, and beta.

    Then touch a `.sentinel` file to signal successful completion.
    """
    if not ctx.default_map:
        click.echo(
            "Please supply a non-empty JSON configuration file via the --config option."
        )
        return
    prefix = ctx.default_map["prefix"]
    model_path = prefix + ".model"
    ctx.invoke(
        create,
        out_path=model_path,
        **restrict_dict_to_params(ctx.default_map, create),
    )
    ctx.invoke(
        train,
        model_path=model_path,
        **restrict_dict_to_params(ctx.default_map, train),
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
    ge_path = prefix + ".ge.pdf"
    ctx.invoke(
        geplot,
        model_path=model_path,
        out=ge_path,
        **restrict_dict_to_params(ctx.default_map, geplot),
    )
    beta_path = prefix + ".beta.pdf"
    ctx.invoke(
        beta,
        model_path=model_path,
        out=beta_path,
        **restrict_dict_to_params(ctx.default_map, beta),
    )
    svd_path = prefix + ".svd.pdf"
    ctx.invoke(
        svd,
        model_path=model_path,
        out=svd_path,
        **restrict_dict_to_params(ctx.default_map, svd),
    )
    heatmap_path = prefix + ".heat.pdf"
    ctx.invoke(
        heatmap,
        model_path=model_path,
        out=heatmap_path,
        **restrict_dict_to_params(ctx.default_map, heatmap),
    )
    sentinel_path = prefix + ".sentinel"
    click.echo(f"LOG: `tdms go` completed; touching {sentinel_path}")
    pathlib.Path(sentinel_path).touch()


@cli.command()
@click.argument("choice_json_path", required=True, type=click.Path(exists=True))
def cartesian(choice_json_path):
    """Take the cartesian product of the variable options in a config file, and
    put it all in an _output directory."""
    make_cartesian_product_hierarchy(from_json_file(choice_json_path))


@cli.command()
@click.argument("source_path", type=click.Path(exists=True))
@click.argument("dest_path", type=click.Path(exists=True))
def transfer(source_path, dest_path):
    """Transfer beta coefficients from one tdms model to another."""
    source_model = torch.load(source_path)
    dest_model = torch.load(dest_path)

    init_weights = source_model.state_dict()["latent_layer.weight"]
    if len(dest_model.latent_layer.weight[0]) != len(init_weights[0]):
        raise ValueError("source & dest beta dimensions do not match.")
    dest_model.latent_layer.weight[0] = init_weights[0]
    dest_model.freeze_betas = True

    torch.save(dest_model, dest_path)
    click.echo(f"LOG: Beta coefficients copied from {source_path} to {dest_path}")


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
