import click
import pandas as pd
import torch

import torchdms.data
import torchdms.model
from torchdms.analysis import Analysis
from torchdms.model import SingleSigmoidNet


@click.group()
def cli():
    pass


@cli.command()
@click.argument("in_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path())
def prep(in_path, out_path):
    """
    Prepare data for training by splitting into test and train, partitioning by
    number of substitutions, and saving the corresponding bmappluses to a pickle.
    """
    [aa_func_scores, wtseq] = torchdms.data.from_pickle_file(in_path)
    test_partition, partitioned_train_data = torchdms.data.partition(aa_func_scores)
    test_data = torchdms.data.bmapplus_of_aa_func_scores(test_partition, wtseq)
    train_data_list = [
        torchdms.data.bmapplus_of_aa_func_scores(train_data_partition, wtseq)
        for train_data_partition in partitioned_train_data
    ]
    torchdms.data.to_pickle_file([test_data, train_data_list], out_path)


@cli.command()
@click.argument("model_name")
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path())
def create(model_name, data_path, out_path):
    """
    Create a model.
    """
    known_models = {"SingleSigmoidNet": SingleSigmoidNet}
    if model_name not in known_models:
        raise IOError(model_name + " not known")
    [test_data, _] = torchdms.data.from_pickle_file(data_path)
    model = known_models[model_name](test_data.binarylength)
    torch.save(model, out_path)
    print(model)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("out_prefix", type=click.Path())
@click.option(
    "--epochs",
    default=5,
    show_default=True,
    help="Number of epochs to use for training.",
)
def train(model_path, data_path, out_prefix, epochs):
    """
    Train a model.
    """
    model = torch.load(model_path)
    [_, train_data_list] = torchdms.data.from_pickle_file(data_path)
    analysis = Analysis(model, train_data_list)
    criterion = torch.nn.MSELoss()
    click.echo(f"Starting training for {epochs} epochs:")
    losses = pd.Series(analysis.train(criterion, epochs))
    losses.to_csv(out_prefix + ".loss.csv")
    ax = losses.plot()
    ax.get_figure().savefig(out_prefix + ".loss.svg")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("out_prefix", type=click.Path())
def eval(model_path, data_path, out_prefix):
    """
    Train a model.
    """
    model = torch.load(model_path)
    model.eval()
    [test_data, _] = torchdms.data.from_pickle_file(data_path)
    analysis = Analysis(model, [])
    results = analysis.evaluate(test_data)
    corr = results.corr().iloc[0, 1]
    results["n_aa_substitutions"] = test_data.n_aa_substitutions
    ax = results.plot.scatter(
        x="Observed", y="Predicted", c=results["n_aa_substitutions"], cmap="viridis"
    )
    ax.text(0, 0.95 * max(results["Predicted"]), f"corr = {corr:.3f}")
    ax.get_figure().savefig(out_prefix + ".scatter.svg")
    print(f"correlation = {corr}")


if __name__ == "__main__":
    cli()
