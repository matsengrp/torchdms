"""
Tools for handling data.
"""

import pickle
import torch
from dms_variants.binarymap import BinaryMap
from torch.utils.data import Dataset


class BinaryMapDataset(Dataset):
    """
    Binarymap dataset.

    This class organizes the information from the input dataset
    into a wrapper containing all relevent attributes for training
    and evaluation. 

    We also store the original dataframe as it may contain
    important metadata (such as target variance), but 
    drop redundant columns that are already attributes
    """

    def __init__(self, pd_dataset, wtseq, targets):

        bmap = BinaryMap(
            pd_dataset.loc[:, ["aa_substitutions"]], expand=True, wtseq=wtseq
        )
        self.samples = torch.from_numpy(bmap.binary_variants.toarray()).float()
        self.targets = torch.from_numpy(pd_dataset[targets].to_numpy()).float()
        self.original_df = pd_dataset.drop(targets, axis=1)
        self.wtseq = wtseq
        self.target_names = targets

    def __getitem__(self, idxs):
        return {"samples": self.samples[idxs], "targets": self.targets[idxs]}

    def __len__(self):
        return self.samples.shape[0]

    def feature_count(self):
        return self.samples.shape[1]


def partition(
    aa_func_scores,
    per_stratum_variants_for_test=100,
    skip_stratum_if_count_is_smaller_than=250,
):
    """
    Partition the data into a test partition, and a list of training data partitions.
    A "stratum" is a slice of the data with a given number of mutations.
    We group training data sets into strata based on their number of mutations so that
    the data is presented the neural network with an even propotion of each.
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
        to_put_in_test = labeled_examples.sample(n=per_stratum_variants_for_test).index
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
    return test_partition, partitioned_train_data


def prepare(test_partition, train_partition_list, wtseq, targets):
    """
    Prepare data for training by splitting into test and train, partitioning by
    number of substitutions, and making bmappluses.
    """

    test_data = BinaryMapDataset(test_partition, wtseq=wtseq, targets=targets)
    train_data_list = [
        BinaryMapDataset(train_data_partition, wtseq=wtseq, targets=targets)
        for train_data_partition in train_partition_list
    ]

    return test_data, train_data_list
