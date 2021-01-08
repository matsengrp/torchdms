"""
Testing for helper methods in analysis.py
"""
import numpy as np
import torch
from torchdms.analysis import low_rank_approximation

def test_low_rank_approximation():
    """ Tests low-rank approximation function."""
    # define simple 2-rank matrix
    test_matrix = np.array([[1,4,7], [2,5,8], [3,6,9]], dtype='float')

    # store true 1-rank approximation here & flatten column-wise
    approx_true = np.array([[1.736218,4.207153,6.678088],
        [2.071742,5.020186,7.968631],
        [2.407267,5.833220,9.259173]]).flatten('F')

    # take low-rank (1) approximation
    approx_est = low_rank_approximation(test_matrix, 1)
    # assert that values match up
    assert torch.allclose(torch.from_numpy(approx_true), approx_est, rtol=0.001)
