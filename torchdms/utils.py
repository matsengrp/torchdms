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
import torchdms.model
import numpy as np
import dms_variants as dms
import seaborn as sns


def from_pickle_file(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def to_pickle_file(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def monotonic_params_from_latent_space(model: torchdms.model.DMSFeedForwardModel):
    """
        following the hueristic that the input layer of a network
        is named 'input_layer' and the weight bias are denoted:

        layer_name.weight
        layer_name.bias.

        this function returns all the parameters
        to be floored to zero in a monotonic model.
        this is every parameter after the latent space
        excluding bias parameters.
        """
    for name, param in model.named_parameters():
        parse_name = name.split(".")
        is_input_layer = parse_name[0] == "input_layer"
        is_bias = parse_name[1] == "bias"
        if not is_input_layer and not is_bias:
            yield param


def evaluatation_dict(model, test_data, device="cpu"):
    """
    Evaluate & Organize all testing data paried with metadata.

    A function which takes a trained model, matching test
    dataset (BinaryMapDataset w/ the same input dimensions.)
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
    width = 7 * num_targets
    fig, ax = plt.subplots(1, num_targets, figsize=(width, 6))
    n_aa_substitutions = [
        len(s.split()) for s in evaluation_dict["original_df"]["aa_substitutions"]
    ]
    for target in range(num_targets):
        pred = evaluation_dict["predictions"][:, target]
        targ = evaluation_dict["targets"][:, target]
        corr = stats.pearsonr(pred, targ)
        if num_targets == 1:
            scatter = ax.scatter(pred, targ, cmap=cmap, c=n_aa_substitutions, s=8.0)
            ax.set_xlabel(f"Predicted")
            ax.set_ylabel(f"Observed")
            target_name = evaluation_dict["target_names"][target]
            ax.set_title(f"Test Data for {target_name}\npearsonr = {round(corr[0],3)}")
        else:
            scatter = ax[target].scatter(
                pred, targ, cmap=cmap, c=n_aa_substitutions, s=8.0
            )
            ax[target].set_xlabel(f"Predicted")
            ax[target].set_ylabel(f"Observed")
            target_name = evaluation_dict["target_names"][target]
            plot_title = f"Test Data for {target_name}\npearsonr = {round(corr[0],3)}"
            ax[target].set_title(plot_title)
            print(plot_title)

    if num_targets == 1:
        ax.legend(
            *scatter.legend_elements(), bbox_to_anchor=(-0.20, 1), title="n-mutant"
        )
    else:
        ax[0].legend(
            *scatter.legend_elements(), bbox_to_anchor=(-0.20, 1), title="n-mutant"
        )
    fig.savefig(out)


def latent_space_contour_plot_2D(model, out, start=0, end=1000, nticks=100):
    """
    This function takes in an Object of type torch.model.DMSFeedForwardModel.
    It uses the `from_latent()` Method to produce a matrix X of predictions given
    combinations or parameters (X_{i}_{j}) fed into the latent space of the model.
    """

    num_targets = model.output_size
    prediction_matrices = [np.empty([nticks, nticks]) for _ in range(num_targets)]
    for i, latent1_value in enumerate(np.linspace(start, end, nticks)):
        for j, latent2_value in enumerate(np.linspace(start, end, nticks)):
            lat_sample = torch.from_numpy(
                np.array([latent1_value, latent2_value])
            ).float()
            predictions = model.from_latent(lat_sample)
            for pred_idx in range(len(predictions)):
                prediction_matrices[pred_idx][i][j] = predictions[pred_idx]

    width = 7 * num_targets
    fig, ax = plt.subplots(1, num_targets, figsize=(width, 6))
    # Make ax a list even if there's only one target.
    if num_targets == 1:
        ax = [ax]
    for idx, matrix in enumerate(prediction_matrices):
        mapp = ax[idx].imshow(matrix)

        # TODO We should have the ticks which show the range of inputs
        # matplotlib does not make this obvious.
        # ax[idx].set_xticks(ticks=np.linspace(start,end,nticks))
        # ax[idx].set_yticks(np.linspace(start,end,nticks))
        ax[idx].set_xlabel("latent space dimension 1")
        ax[idx].set_ylabel("latent space dimension 2")
        ax[idx].set_title(f"Prediction Node {idx}\nrange {start} to {end}")
        fig.colorbar(mapp, ax=ax[idx], shrink=0.5)
    fig.tight_layout()
    fig.savefig(out)


def beta_coefficients(model, test_data, out):
    """
    This function takes in a (ideally trained) model
    and plots the values of the weights corresponding to
    the inputs, dubbed "beta coefficients". We plot this
    as a heatmap where the rows are substitutions nucleotides,
    and the columns are the sequence positions for each mutation.
    """

    # below gives us the first transformation matrix of the model
    # going from inputs -> latent space, thus
    # a tensor of shape (n latent space dims, n input nodes)
    beta_coefficients = next(model.parameters()).data
    bmap = dms.binarymap.BinaryMap(test_data.original_df,)

    # To represent the wtseq in the heatmap, create a mask
    # to encode which matrix entries are the wt nt in each position.
    wtmask = np.full([len(bmap.alphabet), len(test_data.wtseq)], False, dtype=bool)
    alphabet = bmap.alphabet
    for column_position, nt in enumerate(test_data.wtseq):
        row_position = alphabet.index(nt)
        wtmask[row_position, column_position] = True

    # plot beta's
    num_latent_dims = beta_coefficients.shape[0]
    fig, ax = plt.subplots(2, figsize=(10, 5 * num_latent_dims))
    for latent_dim in range(num_latent_dims):
        latent = beta_coefficients[latent_dim].numpy()
        beta_map = latent.reshape(len(bmap.alphabet), len(test_data.wtseq))
        beta_map[wtmask] = np.nan
        mapp = ax[latent_dim].imshow(beta_map, aspect="auto")
        fig.colorbar(mapp, ax=ax[latent_dim], orientation="horizontal")
        ax[latent_dim].set_title(f"Beta coeff for latent dimension {latent_dim}")
        ax[latent_dim].set_yticks(ticks=range(0, 21))
        ax[latent_dim].set_yticklabels(alphabet)
    plt.tight_layout()
    fig.savefig(f"{out}")
