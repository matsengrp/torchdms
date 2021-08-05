"""
Testing for helper methods in analysis.py
"""
import numpy as np
import torch
import os
import pkg_resources
from torchdms.analysis import Analysis
from torchdms.analysis import low_rank_approximation
from torchdms.utils import (
    from_pickle_file,
    to_pickle_file,
)
from torchdms.loss import l1
from torchdms.model import FullyConnected, model_of_string
from torchdms.data import partition, SplitDataset, prep_by_stratum_and_export

TEST_DATA_PATH = pkg_resources.resource_filename("torchdms", "data/test_df.pkl")
out_path = "test_df-prepped"
data_path = out_path + ".pkl"
model_path = "run.model"


def test_low_rank_approximation():
    """Tests low-rank approximation function."""
    # define simple 2-rank matrix
    test_matrix = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]], dtype="float")

    # store true 1-rank approximation here & flatten column-wise
    approx_true = np.array(
        [
            [1.736218, 4.207153, 6.678088],
            [2.071742, 5.020186, 7.968631],
            [2.407267, 5.833220, 9.259173],
        ]
    ).flatten("F")

    # take low-rank (1) approximation
    approx_est = low_rank_approximation(test_matrix, 1)
    # assert that values match up
    assert torch.allclose(torch.from_numpy(approx_true), approx_est, rtol=0.001)


def test_zeroed_wt_betas():
    """Test to ensure WT betas of a model are (and remain) 0."""
    data, wtseq = from_pickle_file(TEST_DATA_PATH)
    split_df = partition(
        data,
        per_stratum_variants_for_test=10,
        skip_stratum_if_count_is_smaller_than=30,
        export_dataframe=None,
        partition_label=None,
    )
    prep_by_stratum_and_export(
        split_df, wtseq, ["affinity_score"], out_path, "", None, protein_start_site=1
    )

    split_df_prepped = from_pickle_file(data_path)

    model_string = "FullyConnected(1,identity,10,relu)"
    model = model_of_string(model_string, data_path)

    torch.save(model, model_path)
    analysis_params = {
        "model": model,
        "model_path": model_path,
        "val_data": split_df_prepped.val,
        "train_data_list": split_df_prepped.train,
    }
    training_params = {"epoch_count": 1, "loss_fn": l1}
    analysis = Analysis(**analysis_params)
    wt_idxs = analysis.val_data.wt_idxs
    assert analysis.val_data.wtseq == "NIT"
    # Assert that wt betas are 0 upon initializaiton.
    for latent_dim in range(analysis.model.latent_dim):
        for idx in wt_idxs:
            assert analysis.model.beta_coefficients()[latent_dim, int(idx)] == 0
    # Train model with analysis object for 1 epoch
    analysis.train(**training_params)

    # Assert that wt betas are still 0.
    for latent_dim in range(analysis.model.latent_dim):
        for idx in wt_idxs:
            assert analysis.model.beta_coefficients()[latent_dim, int(idx)] == 0
    os.remove(model_path)
    os.remove(data_path)
