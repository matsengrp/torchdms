import click
import pandas as pd
import torch

import torchdms.data
import torchdms.model
from torchdms.analysis import Analysis
from torchdms.model import SingleSigmoidNet
from torchdms.model import TwoByOneNet
from torchdms.model import TwoByTwoNet


@click.group()
def cli():
    pass


@cli.command()
@click.argument("in_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path())
def prep(in_path, out_path):
    """
    Prepare data for training. See data.prepare_data for details.
    """
    torchdms.data.to_pickle_file(
        torchdms.data.prepare(*torchdms.data.from_pickle_file(in_path)), out_path
    )


@cli.command()
@click.argument("model_name")
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path())
def create(model_name, data_path, out_path):
    """
    Create a model.
    """
    known_models = {
        "SingleSigmoidNet": SingleSigmoidNet,
        "TwoByOneNet": TwoByOneNet,
        "TwoByTwoNet": TwoByTwoNet,
    }
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
    "--batch-size", default=5000, show_default=True, help="Batch size for training.",
)
@click.option(
    "--learning-rate", default=1e-3, show_default=True, help="Initial learning rate.",
)
@click.option(
    "--epochs", default=5, show_default=True, help="Number of epochs for training.",
)
def train(model_path, data_path, out_prefix, batch_size, learning_rate, epochs):
    """
    Train a model, saving trained model to original location.
    """
    model = torch.load(model_path)
    [_, train_data_list] = torchdms.data.from_pickle_file(data_path)
    analysis = Analysis(
        model, train_data_list, batch_size=batch_size, learning_rate=learning_rate
    )
    criterion = torch.nn.MSELoss()
    training_dict = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    click.echo(f"Starting training. {training_dict}")
    losses = pd.Series(analysis.train(criterion, epochs))
    torch.save(model, model_path)
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
    results["n_aa_substitutions"] = test_data.n_aa_substitutions
    corr, ax = analysis.process_evaluation(results)
    ax.get_figure().savefig(out_prefix + ".scatter.svg")
    print(f"correlation = {corr}")


if __name__ == "__main__":
    cli()
