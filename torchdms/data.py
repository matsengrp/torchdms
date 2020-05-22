"""Tools for handling data."""
from collections import defaultdict
import itertools
import random
import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dms_variants.binarymap import BinaryMap
from torchdms.utils import (
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
    drop redundant columns that are already attributes
    """

    def __init__(self, samples, targets, original_df, wtseq, target_names):
        row_count = len(samples)
        assert row_count == len(targets)
        assert row_count == len(original_df)
        assert targets.shape[1] == len(target_names)
        self.samples = samples
        self.targets = targets
        self.original_df = original_df
        self.wtseq = wtseq
        self.target_names = target_names

    @classmethod
    def of_raw(cls, pd_dataset, wtseq, targets):
        bmap = BinaryMap(
            pd_dataset.loc[:, ["aa_substitutions"]], expand=True, wtseq=wtseq
        )
        return cls(
            torch.from_numpy(bmap.binary_variants.toarray()).float(),
            torch.from_numpy(pd_dataset[targets].to_numpy()).float(),
            pd_dataset.drop(targets, axis=1),
            wtseq,
            targets,
        )

    @classmethod
    def cat(cls, datasets):
        assert isinstance(datasets, list)
        return cls(
            torch.cat([dataset.samples for dataset in datasets], dim=0),
            torch.cat([dataset.targets for dataset in datasets], dim=0),
            pd.concat([dataset.original_df for dataset in datasets]),
            get_only_entry_from_constant_list([dataset.wtseq for dataset in datasets]),
            get_only_entry_from_constant_list(
                [dataset.target_names for dataset in datasets]
            ),
        )

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


def partition(
    aa_func_scores,
    per_stratum_variants_for_test=100,
    skip_stratum_if_count_is_smaller_than=250,
    export_dataframe=None,
    split_label=None,
):
    """Partition the data into a test partition, and a list of training data
    partitions.

    A "stratum" is a slice of the data with a given number of mutations.
    We group training data sets into strata based on their number of
    mutations so that the data is presented the neural network with an
    even propotion of each.
    """
    aa_func_scores["n_aa_substitutions"] = [
        len(s.split()) for s in aa_func_scores["aa_substitutions"]
    ]
    aa_func_scores["in_test"] = False
    partitioned_train_data = []

    for mutation_count, grouped in aa_func_scores.groupby("n_aa_substitutions"):
        if mutation_count == 0:
            continue
        labeled_examples = grouped.dropna()
        if len(labeled_examples) < skip_stratum_if_count_is_smaller_than:
            continue

        # Here, we grab a subset of unique variants so that
        # we are not training on the same variants that we see in the testing data
        unique_variants = defaultdict(list)
        for index, sub in zip(
            labeled_examples.index, labeled_examples["aa_substitutions"]
        ):
            unique_variants[sub].append(index)
        test_variants = random.sample(
            unique_variants.keys(), per_stratum_variants_for_test
        )
        test_dict = {key: unique_variants[key] for key in test_variants}
        to_put_in_test = list(itertools.chain(*list(test_dict.values())))

        aa_func_scores.loc[to_put_in_test, "in_test"] = True
        partitioned_train_data.append(
            aa_func_scores.loc[
                (aa_func_scores["in_test"] == False)
                & (aa_func_scores["n_aa_substitutions"] == mutation_count)
            ].reset_index(drop=True)
        )

    test_partition = aa_func_scores.loc[aa_func_scores["in_test"] == True,].reset_index(
        drop=True
    )

    if export_dataframe is not None:
        if split_label is not None:
            split_label_filename = make_legal_filename(split_label)
            to_pickle_file(
                aa_func_scores, f"{export_dataframe}_{split_label_filename}.pkl"
            )
        else:
            to_pickle_file(aa_func_scores, f"{export_dataframe}.pkl")

    return test_partition, partitioned_train_data


def prepare(test_partition, train_partition_list, wtseq, targets):
    """Prepare data for training by splitting into test and train, partitioning
    by number of substitutions, and making bmappluses."""

    test_data = BinaryMapDataset.of_raw(test_partition, wtseq=wtseq, targets=targets)
    train_data_list = [
        BinaryMapDataset.of_raw(train_data_partition, wtseq=wtseq, targets=targets)
        for train_data_partition in train_partition_list
    ]

    return test_data, train_data_list


def prep_by_stratum_and_export(
    test_partition, partitioned_train_data, wtseq, targets, out_prefix, split_label=None
):
    """Print number of training examples per stratum and test samples, run
    prepare(), and export to .pkl file with descriptive filename."""

    for train_part in partitioned_train_data:
        num_subs = len(train_part["aa_substitutions"][0].split())
        click.echo(
            f"LOG: There are {len(train_part)} training examples "
            f"for stratum: {num_subs}"
        )

    click.echo(f"LOG: There are {len(test_partition)} test points")
    click.echo(f"LOG: Successfully partitioned data")
    click.echo(f"LOG: preparing binary map dataset")

    if split_label is not None:
        split_label_filename = make_legal_filename(split_label)
        to_pickle_file(
            prepare(test_partition, partitioned_train_data, wtseq, list(targets)),
            f"{out_prefix}_{split_label_filename}.pkl",
        )
    else:
        to_pickle_file(
            prepare(test_partition, partitioned_train_data, wtseq, list(targets)),
            f"{out_prefix}.pkl",
        )
