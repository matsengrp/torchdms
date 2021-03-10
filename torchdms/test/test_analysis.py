"""
Testing for helper methods in analysis.py
"""
import numpy as np
import torch
import pkg_resources
from torchdms.analysis import Analysis
from torchdms.analysis import low_rank_approximation
from torchdms.utils import from_pickle_file
from torchdms.loss import l1
from torchdms.model import FullyConnected, model_of_string

split_data_path = pkg_resources.resource_filename(
    "torchdms", "data/test_df.prepped.pkl"
)
model_path = pkg_resources.resource_filename("torchdms", "data/run.model")


def test_low_rank_approximation():
    """ Tests low-rank approximation function."""
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
    """Test to ensure WT betas of a model are (and remain) 0. """
    model_string = "FullyConnected(1,identity,10,relu)"
    model = model_of_string(model_string, split_data_path)
    data = from_pickle_file(split_data_path)
    wt_idxs = data.val.wt_idxs
    analysis_params = {
        "model": model,
        "model_path": model_path,
        "val_data": data.val,
        "train_data_list": data.train,
    }
    training_params = {"epoch_count": 1, "loss_fn": l1}
    analysis = Analysis(**analysis_params)
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
