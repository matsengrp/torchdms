"""Tools for handling data."""
from collections import defaultdict
import random
import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dms_variants.binarymap import BinaryMap
from torchdms.utils import (
    cat_list_values,
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


class SplitData:
    """BinaryMapDatasets for each of test, validation, and train.

    Train is partitioned into a list of BinaryMapDatasets according to
    the number of mutations.
    """

    def __init__(self, *, test_data, val_data, train_data_list, description_string):
        self.test = test_data
        self.val = val_data
        self.train = train_data_list
        self.description_string = description_string

    @property
    def labeled_splits(self):
        """Returns an iterator on (label, split) pairs."""
        return {
            "test": self.test,
            "val": self.val,
            "train": BinaryMapDataset.cat(self.train),
        }.items()


def partition(
    aa_func_scores,
    per_stratum_variants_for_test=100,
    skip_stratum_if_count_is_smaller_than=300,
    export_dataframe=None,
    split_label=None,
):
    """Partition the data into a test partition, and a list of training data
    partitions.

    A "stratum" is a slice of the data with a given number of mutations.
    We group training data sets into strata based on their number of
    mutations so that the data is presented the neural network with an
    even propotion of each.

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
    partitioned_train_data = []

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

        variants_still_available = set(unique_variants.keys()).difference(test_variants)
        val_variants = random.sample(
            variants_still_available, per_stratum_variants_for_test
        )
        to_put_in_val = cat_list_values(unique_variants, val_variants)
        aa_func_scores.loc[to_put_in_val, "in_val"] = True

        assert not (aa_func_scores["in_test"] & aa_func_scores["in_val"]).any()

        partitioned_train_data.append(
            aa_func_scores.loc[
                (~aa_func_scores["in_test"])
                & (~aa_func_scores["in_val"])
                & (aa_func_scores["n_aa_substitutions"] == mutation_count)
            ].reset_index(drop=True)
        )

    test_partition = aa_func_scores.loc[aa_func_scores["in_test"],].reset_index(
        drop=True
    )
    val_partition = aa_func_scores.loc[aa_func_scores["in_val"],].reset_index(drop=True)

    if export_dataframe is not None:
        if split_label is not None:
            split_label_filename = make_legal_filename(split_label)
            to_pickle_file(
                aa_func_scores, f"{export_dataframe}_{split_label_filename}.pkl"
            )
        else:
            to_pickle_file(aa_func_scores, f"{export_dataframe}.pkl")

    return test_partition, val_partition, partitioned_train_data


def prepare(
    test_partition,
    val_partition,
    train_partition_list,
    wtseq,
    targets,
    description_string,
):
    """Prepare data for training by splitting into test, val, and train,
    partitioning by number of substitutions, and making a SplitData object."""

    test_data = BinaryMapDataset.of_raw(test_partition, wtseq=wtseq, targets=targets)
    val_data = BinaryMapDataset.of_raw(val_partition, wtseq=wtseq, targets=targets)
    train_data_list = [
        BinaryMapDataset.of_raw(train_data_partition, wtseq=wtseq, targets=targets)
        for train_data_partition in train_partition_list
    ]

    return SplitData(
        test_data=test_data,
        val_data=val_data,
        train_data_list=train_data_list,
        description_string=description_string,
    )


def prep_by_stratum_and_export(
    test_partition,
    val_partition,
    partitioned_train_data,
    wtseq,
    targets,
    out_prefix,
    description_string,
    split_label,
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
        out_path = f"{out_prefix}_{split_label_filename}.pkl"
    else:
        out_path = f"{out_prefix}.pkl"

    to_pickle_file(
        prepare(
            test_partition,
            val_partition,
            partitioned_train_data,
            wtseq,
            list(targets),
            description_string,
        ),
        out_path,
    )
