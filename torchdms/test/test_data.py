"""
Testing for data.py.
"""

import random
import pandas as pd
import torch
import pkg_resources
from torchdms.data import partition
from torchdms.utils import (
    count_variants_with_a_mutation_towards_an_aa,
    from_pickle_file,
)

TEST_DATA_PATH = pkg_resources.resource_filename("torchdms", "data/test_df.pkl")
split_data_path = pkg_resources.resource_filename(
    "torchdms", "data/_ignore/test_df.prepped.pkl"
)


def test_partition_is_clean():
    """
    Ensure that the test+validation vs train partitions have no shared variants on the
    amino acid level, over 50 random seeds.
    """
    data, _ = from_pickle_file(TEST_DATA_PATH)
    for seed in range(50):
        random.seed(seed)
        split_df = partition(
            data,
            per_stratum_variants_for_test=10,
            skip_stratum_if_count_is_smaller_than=30,
            export_dataframe=None,
            partition_label=None,
        )
        train = pd.concat(split_df.train)
        assert set(split_df.test["aa_substitutions"]).isdisjoint(
            set(train["aa_substitutions"])
        )
        assert set(split_df.test["aa_substitutions"]).isdisjoint(
            set(split_df.val["aa_substitutions"])
        )


def test_summarize():
    """
    Make sure we are calculating summaries correctly.
    """
    data, _ = from_pickle_file(TEST_DATA_PATH)
    aa_substitutions = data["aa_substitutions"]

    def contains_a_mutation_towards_a_k(variant_string):
        for variant in variant_string.split():
            if variant[-1] == "K":
                return True
        return False

    alt_count = sum(
        [contains_a_mutation_towards_a_k(variant) for variant in aa_substitutions]
    )
    assert alt_count == count_variants_with_a_mutation_towards_an_aa(
        aa_substitutions, "K"
    )


def test_wt_idx():
    """
    Ensure that the indicies for the WT-seq are correct.
    """
    data = from_pickle_file(split_data_path)
    actual_idx = torch.Tensor([11, 28, 58])
    assert data.val.wtseq == "NIT"
    assert torch.all(torch.eq(actual_idx, data.val.wt_idxs))
