"""Tools for handling data."""
from collections import defaultdict
import random
import re
import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dms_variants.binarymap import BinaryMap
from torchdms.plot import plot_exploded_dms_variants_dataframe_summary
from torchdms.utils import (
    cat_list_values,
    count_variants_with_a_mutation_towards_an_aa,
    get_only_entry_from_constant_list,
    make_legal_filename,
    to_pickle_file,
)


class BinaryMapDataset(Dataset):
    """Binarymap dataset.

    This class organizes the information from the input dataset
    into a wrapper containing all relevent attributes for training
    and evaluation.

    We also store the original dataframe as it may contain
    important metadata (such as target variance), but
    drop redundant columns that are already attributes.
    """

    def __init__(self, samples, targets, original_df, wtseq, target_names, alphabet):
        row_count = len(samples)
        assert row_count == len(targets)
        assert row_count == len(original_df)
        assert targets.shape[1] == len(target_names)
        self.samples = samples
        self.targets = targets
        self.original_df = original_df
        self.wtseq = wtseq
        self.target_names = target_names
        self.alphabet = alphabet

    @classmethod
    def of_raw(cls, pd_dataset, wtseq, targets):
        bmap = BinaryMap(pd_dataset, expand=True, wtseq=wtseq)
        # check for concentration column
        if 'concentration' in pd_dataset.columns:
            samples = bmap.binary_variants.toarray()
            concentrations = np.array(pd_dataset['concentration'], ndmin=2).T
            concentration_samples = np.concatenate((samples, concentrations), axis=1)
            return cls(
                torch.from_numpy(concentration_samples).float(),
                torch.from_numpy(pd_dataset[targets].to_numpy()).float(),
                pd_dataset,
                wtseq,
                targets,
                bmap.alphabet,
            )
        # else normal of_raw
        return cls(
            torch.from_numpy(bmap.binary_variants.toarray()).float(),
            torch.from_numpy(pd_dataset[targets].to_numpy()).float(),
            pd_dataset,
            wtseq,
            targets,
            bmap.alphabet,
        )

    @classmethod
    def cat(cls, datasets):
        assert isinstance(datasets, list)
        return cls(
            torch.cat([dataset.samples for dataset in datasets], dim=0),
            torch.cat([dataset.targets for dataset in datasets], dim=0),
            pd.concat([dataset.original_df for dataset in datasets], ignore_index=True),
            get_only_entry_from_constant_list([dataset.wtseq for dataset in datasets]),
            get_only_entry_from_constant_list(
                [dataset.target_names for dataset in datasets]
            ),
            get_only_entry_from_constant_list(
                [dataset.alphabet for dataset in datasets]
            ),
        )

    @property
    def wt_idxs(self):
        alphabet_dict = {letter: idx for idx, letter in enumerate(self.alphabet)}
        wt_idx = [alphabet_dict[aa] for aa in self.wtseq]
        wt_encoding_idx = torch.zeros(
            len(self.wtseq), dtype=torch.int, requires_grad=False
        )
        for site, _ in enumerate(self.wtseq):
            wt_encoding_idx[site] = (site * len(self.alphabet)) + wt_idx[site]
        return wt_encoding_idx

    def __getitem__(self, idxs):
        return {"samples": self.samples[idxs], "targets": self.targets[idxs]}

    def __len__(self):
        return self.samples.shape[0]

    def feature_count(self):
        return self.samples.shape[1]

    def target_count(self):
        return len(self.target_names)

    def target_extrema(self):
        """Return a (min, max) tuple for the value of each target."""
        numpy_targets = self.targets.numpy()
        return [(np.nanmin(column), np.nanmax(column)) for column in numpy_targets.T]

    def concentrations_available(self):
        """ Return true if antibody concentrations are available in data."""
        return 'concentration' in self.original_df.columns


class SplitDataframe:
    """Dataframes for each of test, validation, and train.

    Train is partitioned into a list of dataframes according to the
    number of mutations.
    """

    def __init__(self, *, test_data, val_data, train_data_list):
        self.test = test_data
        self.val = val_data
        self.train = train_data_list


class SplitDataset:
    """BinaryMapDatasets for each of test, validation, and train.

    Train is partitioned into a list of BinaryMapDatasets according to
    the number of mutations.
    """

    def __init__(self, *, test_data, val_data, train_data_list, description_string):
        self.test = test_data
        self.val = val_data
        self.train = train_data_list
        self.description_string = description_string

    @classmethod
    def of_split_df(cls, split_df, wtseq, targets, description_string):
        def our_of_raw(df):
            return BinaryMapDataset.of_raw(df, wtseq=wtseq, targets=targets)

        return cls(
            test_data=our_of_raw(split_df.test),
            val_data=our_of_raw(split_df.val),
            train_data_list=[
                our_of_raw(train_data_partition)
                for train_data_partition in split_df.train
            ],
            description_string=description_string,
        )

    @property
    def labeled_splits(self):
        """Returns an iterator on (label, split) pairs."""
        return {
            "test": self.test,
            "val": self.val,
            "train": BinaryMapDataset.cat(self.train),
        }.items()

    def summarize(self, plot_prefix):
        for label, split in self.labeled_splits:
            if plot_prefix is not None:
                expanded_plot_prefix = f"{plot_prefix}_{label}"
            else:
                expanded_plot_prefix = None
            print(f"summary of the {label} split:")
            summarize_dms_variants_dataframe(split.original_df, expanded_plot_prefix)


def summarize_dms_variants_dataframe(df, plot_prefix):
    print(f"    {len(df)} variants")

    def count_variants(aa):
        return count_variants_with_a_mutation_towards_an_aa(df["aa_substitutions"], aa)

    for aa in "*NP":
        print(f"    {count_variants(aa)} variants with a mutation to '{aa}'")

    if plot_prefix is not None:
        plot_exploded_dms_variants_dataframe_summary(
            explode_dms_variants_dataframe(df), plot_prefix + ".pdf"
        )


def expand_substitutions_into_df(substitution_series):
    """Expand a Series of substitutions into a dataframe showing the wt_AA, the
    site, and the mut_AA."""
    pattern = re.compile(r"([^\d]+)(\d+)([^\d]+)")
    return pd.DataFrame(
        substitution_series.apply(lambda s: list(pattern.match(s).groups())).tolist(),
        columns=["wt_AA", "site", "mut_AA"],
    )


def explode_dms_variants_dataframe(in_df):
    """Make a dataframe that has one row for each mutation of every mutated
    variant, showing the wt_AA, the site, and the mut_AA.

    Other information is duplicated as needed.
    """
    df = in_df.copy()
    df["aa_substitutions"] = df["aa_substitutions"].apply(lambda s: s.split())
    df = df.explode("aa_substitutions")
    df.index.name = "variant_index"
    df.reset_index(inplace=True)
    df = df.loc[
        df["n_aa_substitutions"] > 0,
    ]
    df.reset_index(inplace=True, drop=True)
    return pd.concat([expand_substitutions_into_df(df["aa_substitutions"]), df], axis=1)


def check_onehot_encoding(dataset):
    """Asserts that the tensor onehot encoding we have in the Datasets is the
    same as one that we make ourselves from the strings."""
    alphabet_dict = {letter: idx for idx, letter in enumerate(dataset.alphabet)}
    wt_idxs = [alphabet_dict[aa] for aa in dataset.wtseq]
    site_count = len(dataset.wtseq)
    alphabet_length = len(dataset.alphabet)

    idx_arr_from_onehot = np.array(
        [
            row.reshape(site_count, alphabet_length).nonzero()[1]
            for row in dataset.samples.numpy()
        ]
    )

    def variant_idxs_from_exploded_variant(variant):
        """Makes an array with the amino acid indices for each site."""
        idxs = wt_idxs.copy()
        for mut_idx in variant.index:
            idxs[variant.loc[mut_idx, "site"] - 1] = variant.loc[mut_idx, "mut_AA_idx"]
        return idxs

    exploded = explode_dms_variants_dataframe(dataset.original_df)
    exploded["wt_AA_idx"] = exploded["wt_AA"].apply(alphabet_dict.get)
    exploded["mut_AA_idx"] = exploded["mut_AA"].apply(alphabet_dict.get)
    exploded["site"] = exploded["site"].astype("int")
    idx_arr_from_strings = np.array(
        [
            variant_idxs_from_exploded_variant(variant)
            for _, variant in exploded.groupby("variant_index")
        ]
    )

    sorted_idx_arr_from_strings = np.sort(idx_arr_from_strings, axis=0)
    sorted_idx_arr_from_onehot = np.sort(idx_arr_from_onehot, axis=0)
    if not np.array_equal(sorted_idx_arr_from_strings, sorted_idx_arr_from_onehot):
        raise AssertionError("check_onehot_encoding failed!")


def partition(
    aa_func_scores,
    per_stratum_variants_for_test,
    skip_stratum_if_count_is_smaller_than,
    export_dataframe,
    partition_label,
):
    """Partition the data as needed and build a SplitDataframe.

    A "stratum" is a slice of the data with a given number of mutations. We group
    training data sets into strata based on their number of mutations so that the data
    is presented the neural network with an even proportion of each.

    Furthermore, we group data rows by unique variants and then split on those grouped
    items so that we don't have the same variant showing up in train and test.
    """
    if skip_stratum_if_count_is_smaller_than < 3 * per_stratum_variants_for_test:
        raise IOError(
            "You may have fewer than 3x the number of per_stratum_variants_for_test than "
            "you have in a stratum. Recall that we want to have test, validation, and "
            "train for each stratum."
        )
    aa_func_scores["n_aa_substitutions"] = [
        len(s.split()) for s in aa_func_scores["aa_substitutions"]
    ]
    aa_func_scores["in_test"] = False
    aa_func_scores["in_val"] = False
    test_split_strata = []

    for mutation_count, grouped in aa_func_scores.groupby("n_aa_substitutions"):
        if mutation_count == 0:
            continue
        labeled_examples = grouped.dropna()
        if len(labeled_examples) < skip_stratum_if_count_is_smaller_than:
            continue

        # Here, we grab a subset of unique variants so that we are not training on the
        # same variants that we see in the testing data.
        unique_variants = defaultdict(list)
        for index, sub in zip(
            labeled_examples.index, labeled_examples["aa_substitutions"]
        ):
            unique_variants[sub].append(index)

        test_variants = random.sample(
            unique_variants.keys(), per_stratum_variants_for_test
        )
        to_put_in_test = cat_list_values(unique_variants, test_variants)
        aa_func_scores.loc[to_put_in_test, "in_test"] = True

        # We convert this set difference into a sorted list, because sets have an
        # unspecified order in Python. This is important to get deterministic behavior
        # when we set the seed.
        variants_still_available = sorted(
            set(unique_variants.keys()).difference(test_variants)
        )
        val_variants = random.sample(
            variants_still_available, per_stratum_variants_for_test
        )
        to_put_in_val = cat_list_values(unique_variants, val_variants)
        aa_func_scores.loc[to_put_in_val, "in_val"] = True

        assert not (aa_func_scores["in_test"] & aa_func_scores["in_val"]).any()

        test_split_strata.append(
            aa_func_scores.loc[
                (~aa_func_scores["in_test"])
                & (~aa_func_scores["in_val"])
                & (aa_func_scores["n_aa_substitutions"] == mutation_count)
            ].reset_index(drop=True)
        )

    test_split = aa_func_scores.loc[
        aa_func_scores["in_test"],
    ].reset_index(drop=True)
    val_split = aa_func_scores.loc[
        aa_func_scores["in_val"],
    ].reset_index(drop=True)

    if export_dataframe is not None:
        if partition_label is not None:
            partition_label_filename = make_legal_filename(partition_label)
            to_pickle_file(
                aa_func_scores, f"{export_dataframe}_{partition_label_filename}.pkl"
            )
        else:
            to_pickle_file(aa_func_scores, f"{export_dataframe}.pkl")

    return SplitDataframe(
        test_data=test_split,
        val_data=val_split,
        train_data_list=test_split_strata,
    )


def prep_by_stratum_and_export(
    split_df,
    wtseq,
    targets,
    out_prefix,
    description_string,
    partition_label,
):
    """Print number of training examples per stratum and test samples, run
    prepare(), and export to .pkl file with descriptive filename."""

    for train_part in split_df.train:
        num_subs = len(train_part["aa_substitutions"][0].split())
        click.echo(
            f"LOG: There are {len(train_part)} training samples "
            f"for stratum: {num_subs}"
        )

    click.echo(f"LOG: There are {len(split_df.val)} validation samples")
    click.echo(f"LOG: There are {len(split_df.test)} test samples")
    click.echo("LOG: Successfully partitioned data")
    click.echo("LOG: preparing binary map dataset")

    if partition_label is not None:
        partition_label_filename = make_legal_filename(partition_label)
        out_path = f"{out_prefix}_{partition_label_filename}.pkl"
    else:
        out_path = f"{out_prefix}.pkl"

    to_pickle_file(
        SplitDataset.of_split_df(
            split_df,
            wtseq,
            list(targets),
            description_string,
        ),
        out_path,
    )
