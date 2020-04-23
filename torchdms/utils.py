import click
import scipy.stats as stats
import itertools
import pandas as pd
import torch
import pickle
from torch.utils.data import DataLoader
from torchdms.data import BinaryMapDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def from_pickle_file(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def to_pickle_file(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def evaluatation_dict(model, test_data, device="cpu"):
    """
    Evaluate & Organize all testing data paried with metadata.

    A function which takes a trained model, matching test 
    dataset (BinaryMapDataset w/ the same input dimentions.)
    and return a dictionary containing the 

    - samples: binary encodings numpy array shape (num samples, num possible mutations)
    - predictions and targets: both numpy arrays of shape (num samples, num targets)
    -

    This should have everything mostly needed to do plotting
    about testing data (not things like loss or latent space prediction)
    """
    # TODO check the testing dataset matches the
    # model input size and output size!

    return {
        "samples": test_data.samples.detach().numpy(),
        "predictions": model(test_data.samples.to(device)).detach().numpy(),
        "targets": test_data.targets.detach().numpy(),
        "original_df": test_data.original_df,
        "wtseq": test_data.wtseq,
        "target_names": test_data.target_names,
    }


def plot_test_correlation(evaluation_dict, out, cmap="plasma"):
    """
    Plot scatter plot and correlation values between predicted and 
    observed for each target
    """
    num_targets = evaluation_dict["targets"].shape[1]
    fig, ax = plt.subplots(1, num_targets, figsize=(15, 6))
    n_aa_substitutions = [
        len(s.split()) for s in evaluation_dict["original_df"]["aa_substitutions"]
    ]
    for target in range(num_targets):
        pred = evaluation_dict["predictions"][:, target]
        targ = evaluation_dict["targets"][:, target]
        corr = stats.pearsonr(pred, targ)
        scatter = ax[target].scatter(pred, targ, cmap=cmap, c=n_aa_substitutions, s=8.0)
        ax[target].set_xlabel(f"Predicted")
        ax[target].set_ylabel(f"Observed")
        target_name = evaluation_dict["target_names"][target]
        ax[target].set_title(
            f"Test Data for {target_name}\npearsonr = {round(corr[0],3)}"
        )
    ax[0].legend(
        *scatter.legend_elements(), bbox_to_anchor=(-0.20, 1), title="n-mutant"
    )
    fig.savefig(out)
