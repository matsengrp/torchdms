"""
Tools for handling data.
"""

import pickle
import torch
from dms_variants.binarymap import BinaryMap
from torch.utils.data import Dataset


def from_pickle_file(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def to_pickle_file(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


class BinarymapDataset(Dataset):
    """Binarymap dataset."""

    def __init__(self, bmap):
        self.binary_variants_array = bmap.binary_variants.toarray()
        self.bmap = bmap
        self.variants = torch.from_numpy(self.binary_variants_array).float()
        self.func_scores = torch.from_numpy(self.bmap.func_scores).float()

    def __len__(self):
        return self.bmap.nvariants

    def __getitem__(self, idxs):
        return {"variants": self.variants[idxs], "func_scores": self.func_scores[idxs]}

    def feature_count(self):
        return self.binary_variants_array.shape[1]


def partition(
    aa_func_scores,
    per_stratum_variants_for_test=250,
    skip_stratum_if_count_is_smaller_than=500,
):
    """
    Partition the data into a test partition, and a list of training data partitions.
    """
    aa_func_scores["n_aa_substitutions"] = [
        len(s.split()) for s in aa_func_scores["aa_substitutions"]
    ]
    aa_func_scores["in_test"] = False
    partitioned_train_data = []

    for mutation_count, grouped in aa_func_scores.groupby("n_aa_substitutions"):
        if len(grouped) < skip_stratum_if_count_is_smaller_than:
            continue
        to_put_in_test = grouped.sample(n=per_stratum_variants_for_test).index
        aa_func_scores.loc[to_put_in_test, "in_test"] = True
        partitioned_train_data.append(
            aa_func_scores.loc[
                (aa_func_scores["in_test"] == False)
                & (aa_func_scores["n_aa_substitutions"] == mutation_count)
            ]
        )

    test_partition = aa_func_scores.loc[
        aa_func_scores["in_test"] == True,
    ]
    return test_partition, partitioned_train_data


def bmapplus_of_aa_func_scores(aa_func_scores_subset, wtseq):
    """
    Define a "bmapplus" to be a BinaryMap but where we also have `aa_substitutions` and
    `n_aa_substitutions` data attributes.

    This function makes a bamapplus out of an aa_func_scores dataframe, which may be
    subset from another.
    """
    aa_func_scores_standalone = aa_func_scores_subset.reset_index(drop=True)
    bmap = BinaryMap(aa_func_scores_standalone, expand=True, wtseq=wtseq)
    bmap.aa_substitutions = aa_func_scores_standalone["aa_substitutions"]
    bmap.n_aa_substitutions = aa_func_scores_standalone["n_aa_substitutions"]
    return bmap
