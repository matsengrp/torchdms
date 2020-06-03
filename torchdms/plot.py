"""Plotting functions."""

import os.path
import warnings
import dms_variants as dms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import (
    aes,
    geom_point,
    geom_smooth,
    ggplot,
    ggtitle,
    save_as_pdf_pages,
    theme_seaborn,
    theme_set,
)
import scipy.stats as stats


def plot_error(error_df, out_path, show_points=False):
    theme_set(theme_seaborn(style="whitegrid", context="paper"))
    base_plot = [
        aes(x="observed", y="abs_error", color="n_aa_substitutions"),
        geom_smooth(aes(group="n_aa_substitutions"), method="lowess"),
    ]
    if show_points:
        base_plot.append(geom_point(alpha=0.1))
    warnings.filterwarnings("ignore")
    plots = [
        ggplot(per_target_error_df) + base_plot + ggtitle(target)
        for target, per_target_error_df in error_df.groupby("target")
    ]
    save_as_pdf_pages(plots, filename=out_path, verbose=False)


def plot_test_correlation(evaluation_dict, model, out, cmap="plasma"):
    """Plot scatter plot and correlation values between predicted and observed
    for each target."""
    num_targets = evaluation_dict["targets"].shape[1]
    n_aa_substitutions = [
        len(s.split()) for s in evaluation_dict["original_df"]["aa_substitutions"]
    ]
    width = 7 * num_targets
    fig, ax = plt.subplots(1, num_targets, figsize=(width, 6))
    if num_targets == 1:
        ax = [ax]
    correlation_series = {}
    for target in range(num_targets):
        pred = evaluation_dict["predictions"][:, target]
        targ = evaluation_dict["targets"][:, target]
        corr = stats.pearsonr(pred, targ)
        scatter = ax[target].scatter(pred, targ, cmap=cmap, c=n_aa_substitutions, s=8.0)
        ax[target].set_xlabel("Predicted")
        ax[target].set_ylabel("Observed")
        target_name = evaluation_dict["target_names"][target]
        plot_title = f"Test Data for {target_name}\npearsonr = {round(corr[0],3)}"
        ax[target].set_title(plot_title)
        print(plot_title)

        per_target_df = pd.DataFrame(
            dict(pred=pred, targ=targ, n_aa_substitutions=n_aa_substitutions,)
        )
        correlation_series["correlation " + str(target)] = (
            per_target_df.groupby("n_aa_substitutions").corr().iloc[0::2, -1]
        )

    correlation_df = pd.DataFrame(correlation_series)
    correlation_df.index = correlation_df.index.droplevel(1)
    correlation_path = os.path.splitext(out)[0]
    correlation_df["model"] = model.str_summary()
    for name, characteristic in model.characteristics.items():
        correlation_df[name] = characteristic
    correlation_df.to_csv(correlation_path + ".corr.csv")

    ax[0].legend(
        *scatter.legend_elements(), bbox_to_anchor=(-0.20, 1), title="n-mutant"
    )
    fig.savefig(out)


def latent_space_contour_plot_2d(model, out, start=0, end=1000, nticks=100):
    """This function takes in an Object of type torch.model.VanillaGGE.

    It uses the `from_latent()` Method to produce a matrix X of
    predictions given combinations or parameters (X_{i}_{j}) fed into
    the latent space of the model.
    """

    raise NotImplementedError(
        "This is currently broken, with the idea of reimplementing "
        "it like https://github.com/matsengrp/torchdms/issues/26"
    )


def beta_coefficients(model, test_data, out):
    """This function takes in a (ideally trained) model and plots the values of
    the weights corresponding to the inputs, dubbed "beta coefficients".

    We plot this as a heatmap where the rows are substitutions
    nucleotides, and the columns are the sequence positions for each
    mutation.
    """

    # below gives us the first transformation matrix of the model
    # going from inputs -> latent space, thus
    # a tensor of shape (n latent space dims, n input nodes)
    beta_coefficient_data = next(model.parameters()).data
    bmap = dms.binarymap.BinaryMap(test_data.original_df,)

    # To represent the wtseq in the heatmap, create a mask
    # to encode which matrix entries are the wt nt in each position.
    wtmask = np.full([len(bmap.alphabet), len(test_data.wtseq)], False, dtype=bool)
    alphabet = bmap.alphabet
    for column_position, aa in enumerate(test_data.wtseq):
        row_position = alphabet.index(aa)
        wtmask[row_position, column_position] = True

    # plot beta's
    num_latent_dims = beta_coefficient_data.shape[0]
    fig, ax = plt.subplots(num_latent_dims, figsize=(10, 5 * num_latent_dims))
    if num_latent_dims == 1:
        ax = [ax]
    for latent_dim in range(num_latent_dims):
        latent = beta_coefficient_data[latent_dim].numpy()
        beta_map = latent.reshape(len(bmap.alphabet), len(test_data.wtseq))
        beta_map[wtmask] = np.nan
        mapp = ax[latent_dim].imshow(beta_map, aspect="auto")
        fig.colorbar(mapp, ax=ax[latent_dim], orientation="horizontal")
        ax[latent_dim].set_title(f"Beta coeff for latent dimension {latent_dim}")
        ax[latent_dim].set_yticks(ticks=range(0, 21))
        ax[latent_dim].set_yticklabels(alphabet)
    plt.tight_layout()
    fig.savefig(f"{out}")
