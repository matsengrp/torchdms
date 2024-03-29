"""Plotting functions."""

import math
import os.path
import warnings
import binarymap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import colors, patches
from plotnine import (
    aes,
    facet_grid,
    geom_bar,
    geom_line,
    geom_point,
    geom_density,
    geom_smooth,
    geom_tile,
    ggplot,
    ggtitle,
    save_as_pdf_pages,
    theme,
    theme_seaborn,
    theme_set,
    theme_void,
)
from scipy import stats
from torchdms.utils import build_beta_map


def plot_exploded_binarymap_dataframe_summary(exploded_df, out_path):
    plots = [
        (
            ggplot(exploded_df, aes("site"))
            + geom_bar()
            + facet_grid("mut_AA~")
            + theme_void()
            + theme(figure_size=(10, 8))
            + ggtitle("distribution of mutant positions")
        ),
        (
            ggplot(exploded_df, aes("affinity_score"))
            + geom_density(fill="steelblue")
            + facet_grid("mut_AA~", scales="free_y")
            + theme_void()
            + theme(figure_size=(10, 8))
            + ggtitle("distribution of affinity score")
        ),
    ]
    save_as_pdf_pages(plots, filename=out_path, verbose=False)


def plot_error(error_df, out_path, title_prefix, show_points=False):
    theme_set(theme_seaborn(style="whitegrid", context="paper"))
    base_plot = [
        aes(x="observed", y="abs_error", color="n_aa_substitutions"),
        geom_smooth(aes(group="n_aa_substitutions"), method="lowess"),
    ]
    if show_points:
        base_plot.append(geom_point(alpha=0.1))
    warnings.filterwarnings("ignore")
    plots = [
        ggplot(per_target_error_df) + base_plot + ggtitle(f"{target}: {title_prefix}")
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
    has_unseen_mutation = np.array(evaluation_dict["original_df"]["unseen_mutations"])
    width = 7 * num_targets
    fig, ax = plt.subplots(1, num_targets, figsize=(width, 6))
    if num_targets == 1:
        ax = [ax]
    correlation_series = {}
    for target in range(num_targets):
        pred = evaluation_dict["predictions"][:, target]
        targ = evaluation_dict["targets"][:, target]
        predc = pred[~np.isnan(targ)]
        targc = targ[~np.isnan(targ)]
        corr = stats.pearsonr(predc, targc)
        # create plot with seen mutations and then unseen mutations
        scatter = ax[target].scatter(
            np.array(pred)[~has_unseen_mutation],
            np.array(targ)[~has_unseen_mutation],
            cmap=cmap,
            c=np.array(n_aa_substitutions)[~has_unseen_mutation],
            marker=".",
            s=8.0,
        )
        ax[target].scatter(
            np.array(pred)[has_unseen_mutation],
            np.array(targ)[has_unseen_mutation],
            cmap=cmap,
            c=np.array(n_aa_substitutions)[has_unseen_mutation],
            marker="x",
            s=32.0,
            alpha=0.8,
        )
        ax[target].set_xlabel("Predicted")
        ax[target].set_ylabel("Observed")
        target_name = evaluation_dict["target_names"][target]
        plot_title = f"{target_name}:"
        ax[target].set_title(plot_title, size=16)

        ax[target].text(
            0.15,
            0.95,
            f"$R^{2}$ = {round(corr[0],3)}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax[target].transAxes,
            size=15,
        )

        per_target_df = pd.DataFrame(
            dict(
                pred=pred,
                targ=targ,
                n_aa_substitutions=n_aa_substitutions,
            )
        )
        correlation_series["correlation " + str(target)] = (
            per_target_df.groupby("n_aa_substitutions").corr().iloc[0::2, -1]
        )

    model_type = model.str_summary().split(":")[0]
    ax[0].text(
        0.5,
        -0.15,
        f"{model_type}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[0].transAxes,
        size=15,
    )

    if len(model.str_summary().split(":")) > 1:

        model_architecture = "\n".join(model.str_summary().split(":")[1].split())[1:-1]

        ax[0].text(
            0.5,
            -0.25,
            f"\n{model_architecture}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax[0].transAxes,
            size=15,
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
    plt.tight_layout()
    fig.savefig(out)


def pretty_breaks(break_count):
    def make_pretty_breaks(passed_breaks):
        return passed_breaks[:: math.ceil(len(passed_breaks) / break_count)]

    return make_pretty_breaks


def plot_heatmap(model, test_data, out):
    """This function takes in a model and plots the single mutant predictions.

    We plot this as a heatmap where the rows are substitutions
    nucleotides, and the columns are the sequence positions for each
    mutation.
    """
    theme_set(theme_seaborn(style="whitegrid", context="paper"))
    predictions = model.single_mutant_predictions()

    bmap = binarymap.BinaryMap(
        test_data.original_df,
    )

    # To represent the wtseq in the heatmap, create a mask
    # to encode which matrix entries are the wt nt in each position.
    wtmask = np.full([len(bmap.alphabet), len(test_data.wtseq)], False, dtype=bool)
    alphabet = bmap.alphabet
    for column_position, aa in enumerate(test_data.wtseq):
        row_position = alphabet.index(aa)
        wtmask[row_position, column_position] = True

    num_prediction_dims = len(predictions)
    fig, ax = plt.subplots(num_prediction_dims, figsize=(10, 5 * num_prediction_dims))
    if num_prediction_dims == 1:
        ax = [ax]
    for prediction_dim in range(num_prediction_dims):
        beta_map = predictions[prediction_dim]

        # define your scale, with white at zero
        mapp = ax[prediction_dim].imshow(
            beta_map, aspect="auto", norm=colors.TwoSlopeNorm(0), cmap="RdBu"
        )
        # Box WT-cells.
        for wt_idx in np.transpose(wtmask.nonzero()):
            wt_cell = patches.Rectangle(
                np.flip(wt_idx - 0.5),
                1,
                1,
                facecolor="none",
                edgecolor="black",
                linewidth=2,
            )
            ax[prediction_dim].add_patch(wt_cell)
        fig.colorbar(mapp, ax=ax[prediction_dim], orientation="horizontal")
        ax[prediction_dim].set_yticks(ticks=range(0, 21))
        ax[prediction_dim].set_yticklabels(alphabet)
    plt.tight_layout()
    fig.suptitle(f"{model.str_summary()}")
    fig.savefig(f"{out}")


def beta_coefficients(model, test_data, out):
    """This function takes in a (ideally trained) model and plots the values of
    the weights corresponding to the inputs, dubbed "beta coefficients".

    We plot this as a heatmap where the rows are substitutions
    nucleotides, and the columns are the sequence positions for each
    mutation.
    """

    bmap = binarymap.BinaryMap(
        test_data.original_df,
    )

    # To represent the wtseq in the heatmap, create a mask
    # to encode which matrix entries are the wt nt in each position.
    wtmask = np.full([len(bmap.alphabet), len(test_data.wtseq)], False, dtype=bool)
    alphabet = bmap.alphabet
    for column_position, aa in enumerate(test_data.wtseq):
        row_position = alphabet.index(aa)
        wtmask[row_position, column_position] = True

    # plot beta's
    num_latent_dims = model.beta_coefficients().shape[0]
    fig, ax = plt.subplots(num_latent_dims, figsize=(10, 5 * num_latent_dims))
    if num_latent_dims == 1:
        ax = [ax]
    for latent_dim in range(num_latent_dims):
        beta_map = build_beta_map(
            test_data.wtseq,
            test_data.alphabet,
            model.beta_coefficients()[latent_dim].numpy(),
        )
        # define your scale, with white at zero
        mapp = ax[latent_dim].imshow(
            beta_map, aspect="auto", norm=colors.TwoSlopeNorm(0), cmap="RdBu"
        )
        # Box WT-cells.
        for wt_idx in np.transpose(wtmask.nonzero()):
            wt_cell = patches.Rectangle(
                np.flip(wt_idx - 0.5),
                1,
                1,
                facecolor="none",
                edgecolor="black",
                linewidth=2,
            )
            ax[latent_dim].add_patch(wt_cell)
        fig.colorbar(mapp, ax=ax[latent_dim], orientation="horizontal")
        ax[latent_dim].set_yticks(ticks=range(0, 21))
        ax[latent_dim].set_yticklabels(alphabet)
    plt.tight_layout()
    fig.suptitle(f"{model.str_summary()}")
    fig.savefig(f"{out}")


def df_with_named_columns_of_np_array(x, column_prefix):
    return pd.DataFrame(
        x, columns=[f"{column_prefix}_{col_idx}" for col_idx in range(x.shape[1])]
    )


def build_geplot_df(model, data, device="cpu"):
    """Build data frame for making a global epistasis plot."""
    assert data.feature_count() == model.input_size
    assert data.target_count() == model.output_size
    model.eval()
    return pd.concat(
        [
            df_with_named_columns_of_np_array(
                model.to_latent(data.samples.to(device)).detach().numpy(), "latent"
            ),
            df_with_named_columns_of_np_array(
                model(data.samples.to(device)).detach().numpy(), "prediction"
            ),
            df_with_named_columns_of_np_array(data.targets.detach().numpy(), "target"),
        ],
        axis=1,
    )


def plot_geplot(geplot_df, path, title):
    theme_set(theme_seaborn(style="ticks", context="paper"))
    alpha = 0.3
    (
        ggplot(geplot_df)
        + geom_point(aes("latent_0", "target_0"), alpha=alpha)
        + geom_line(aes("latent_0", "prediction_0"), color="red")
        + ggtitle(title)
    ).save(path)


def series_min_max(series):
    return series.min(), series.max()


def build_2d_nonlinearity_df(model, geplot_df, steps):
    """Build a dataframe that contains the value of the nonlinearity for the
    domain of the range of values seen in the geplot_df."""
    latent_keys = ["latent_0", "latent_1"]
    latent_domains = [series_min_max(geplot_df[key]) for key in latent_keys]
    inputs = torch.cartesian_prod(
        torch.linspace(*latent_domains[0], steps=steps),
        torch.linspace(*latent_domains[1], steps=steps),
    )
    predictions = df_with_named_columns_of_np_array(
        model.from_latent_to_output(inputs).detach().numpy(), "prediction"
    )
    return pd.concat(
        [pd.DataFrame(inputs.detach().numpy(), columns=latent_keys), predictions],
        axis=1,
    )


def plot_2d_geplot(model, geplot_df, nonlinearity_df, path):
    def make_plot_for(output_idx):
        return (
            ggplot()
            + geom_tile(
                aes("latent_0", "latent_1", fill=f"prediction_{output_idx}"),
                nonlinearity_df,
            )
            + geom_point(
                aes("latent_0", "latent_1", fill=f"target_{output_idx}"),
                geplot_df,
                stroke=0.05,
            )
            + ggtitle(f"{model.target_names[output_idx]}: {model.str_summary()}")
        )

    save_as_pdf_pages(
        [make_plot_for(output_idx) for output_idx in range(len(model.target_names))],
        filename=path,
        verbose=False,
    )


def plot_svd(model, test_data, out):
    """This function plots the log singular values and the cummulative sum of
    each of a trained model's beta coefficent matricies."""

    num_latent_dims = model.beta_coefficients().shape[0]

    fig, ax = plt.subplots(
        nrows=num_latent_dims, ncols=2, figsize=(10, 5 * num_latent_dims)
    )
    for latent_dim in range(num_latent_dims):
        beta_map = build_beta_map(
            test_data.wtseq,
            test_data.alphabet,
            model.beta_coefficients()[latent_dim].numpy(),
        )
        rank = np.linalg.matrix_rank(beta_map)
        s_matrix = np.linalg.svd(beta_map, compute_uv=False)

        sing_vals = np.arange(rank) + 1  # index singular values for plotting
        sing_vals = sing_vals[:rank]
        s_matrix = s_matrix[:rank]
        sing_vals_cumsum = np.cumsum(s_matrix) / np.sum(s_matrix)

        if num_latent_dims > 1:
            ax[latent_dim, 0].plot(sing_vals, np.log10(s_matrix), "ro-", linewidth=2)
            ax[latent_dim, 0].set_xlabel("j")
            ax[latent_dim, 0].set_ylabel(r"$log(\sigma_j)$")
            ax[latent_dim, 0].set_title(
                f"Singular values for {latent_dim}, rank={rank}"
            )

            ax[latent_dim, 1].plot(sing_vals, sing_vals_cumsum, "ro-", linewidth=2)
            ax[latent_dim, 1].set_xlabel("j")
            ax[latent_dim, 1].set_ylabel("Cummulative value %")
            ax[latent_dim, 1].set_title(
                f"Cummulative singular values for {latent_dim}, rank={rank}"
            )
        else:
            ax[0].plot(sing_vals, np.log10(s_matrix), "ro-", linewidth=2)
            ax[0].set(
                xlabel="j",
                ylabel=r"$log(\sigma_j)$",
                title=f"Singular values for {latent_dim}, rank={rank}",
            )
            ax[1].plot(sing_vals, sing_vals_cumsum, "ro-", linewidth=2)
            ax[1].set(
                xlabel="j",
                ylabel="Cummulative value %",
                title=f"Cummulative singular values for {latent_dim}, rank={rank}",
            )

    # plt.tight_layout()
    fig.suptitle(f"{model.str_summary()}")
    fig.savefig(f"{out}")


def plot_svd_profiles(model, test_data, out):
    """Plots heatmaps of amino acid profiles (U) and site profiles (V) from SVD
    output after final gradient step of training."""
    num_latent_dims = model.beta_coefficients().shape[0]
    seq_len = model.sequence_length
    fig, ax = plt.subplots(
        nrows=num_latent_dims, ncols=2, figsize=(10, 5 * num_latent_dims)
    )
    if num_latent_dims == 1:
        latent_dim = 0
        beta_map = build_beta_map(
            test_data.wtseq,
            test_data.alphabet,
            model.beta_coefficients()[latent_dim].numpy(),
        )
        rank = np.linalg.matrix_rank(beta_map)
        u_vecs, _, v_vecs = torch.svd(torch.from_numpy(beta_map))
        # Plot amino acid profiles
        aa_profiles = ax[0].imshow(u_vecs[:, :rank], aspect="auto", cmap="Reds")
        fig.colorbar(aa_profiles, ax=ax[0], orientation="horizontal")
        ax[0].set(
            title=f"Amino acid profiles for latent dim {latent_dim}, rank={rank}",
            xticks=range(rank),
            xlabel="Profile number",
            yticks=range(0, 21),
            yticklabels=test_data.alphabet,
            ylabel="Amino acid",
        )
        # add second heatmap for folding latent space
        site_profiles = ax[1].imshow(v_vecs[:, :rank], aspect="auto", cmap="Reds")
        fig.colorbar(site_profiles, ax=ax[1], orientation="horizontal")
        ax[1].set(
            title=f"Per-site profile usage for latent dim {latent_dim}, rank={rank}",
            xlabel="Profile number",
            xticks=range(rank),
            yticks=range(0, seq_len, 10),
            ylabel="Site number",
        )
    else:
        for latent_dim in range(num_latent_dims):
            beta_map = build_beta_map(
                test_data.wtseq,
                test_data.alphabet,
                model.beta_coefficients()[latent_dim].numpy(),
            )
            rank = np.linalg.matrix_rank(beta_map)
            u_vecs, _, v_vecs = torch.svd(torch.from_numpy(beta_map))
            # Plot amino acid profiles
            aa_profiles = ax[latent_dim, 0].imshow(
                u_vecs[:, :rank], aspect="auto", cmap="Reds"
            )
            fig.colorbar(aa_profiles, ax=ax[latent_dim, 0], orientation="horizontal")
            ax[latent_dim, 0].set(
                title=f"Amino acid profiles, rank={rank}",
                xticks=range(rank),
                xlabel="Profile number",
                yticks=range(0, 21),
                yticklabels=test_data.alphabet,
                ylabel="Amino acid",
            )
            # add second heatmap for folding latent space
            site_profiles = ax[latent_dim, 1].imshow(
                v_vecs[:, :rank], aspect="auto", cmap="Reds"
            )
            fig.colorbar(site_profiles, ax=ax[latent_dim, 1], orientation="horizontal")
            ax[latent_dim, 1].set(
                title=f"Per-site profile usage, rank={rank}",
                xlabel="Profile number",
                xticks=range(rank),
                yticks=range(0, seq_len, 10),
                ylabel="Site number",
            )

    fig.suptitle(f"{model.str_summary()}")
    fig.savefig(f"{out}")


def plot_observed_2d_scores(data, targets, out):
    """If a dataset has two targets, create a scatterplot of the scores."""
    fig, ax = plt.subplots(figsize=(10, 5))
    corr, _ = stats.pearsonr(data[targets[0]], data[targets[1]])
    plt.title(f"pearsonr={round(corr, 3)}")
    plt.suptitle(f"Observed variants {targets[0]} vs {targets[1]}")
    plot = ax.scatter(data[targets[0]], data[targets[1]], c=data["n_aa_substitutions"])
    ax.set_xlabel(targets[0])
    ax.set_ylabel(targets[1])

    ax.legend(*plot.legend_elements(), bbox_to_anchor=(1, 1), title="n-mutant")

    fig.savefig(f"{out}")
