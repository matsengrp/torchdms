"""
Testing for data.py.
"""

import random
import pandas as pd
import pkg_resources
from torchdms.data import partition
from torchdms.utils import from_pickle_file

TEST_DATA_PATH = pkg_resources.resource_filename("torchdms", "data/test_df.pkl")


def test_partition_is_clean():
    """
    Ensure that the test+validation vs train partitions have no shared variants on the
    amino acid level, over 50 random seeds.
    """
    data, _ = from_pickle_file(TEST_DATA_PATH)
    for seed in range(50):
        random.seed(seed)
        (test, val, partitioned_train) = partition(
            data,
            per_stratum_variants_for_test=10,
            skip_stratum_if_count_is_smaller_than=30,
        )
        train = pd.concat(partitioned_train)
        assert set(test["aa_substitutions"]).isdisjoint(set(train["aa_substitutions"]))
        assert set(test["aa_substitutions"]).isdisjoint(set(val["aa_substitutions"]))
