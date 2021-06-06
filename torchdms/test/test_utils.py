"""
Testing for utils.py.
"""
import torch
from torchdms.utils import cartesian_product, affine_projection_matrix, project_betas


def test_cartesian_product():
    """
    Test the cartesian_product function.
    """
    test = cartesian_product({"i": 5})
    correct = [([], {"i": 5})]
    assert correct == test
    test = cartesian_product({"i": [5, 6]})
    correct = [(["i@5"], {"i": 5}), (["i@6"], {"i": 6})]
    test = cartesian_product({"s": "a", "i": [5, 6]})
    correct = [(["i@5"], {"s": "a", "i": 5}), (["i@6"], {"s": "a", "i": 6})]
    assert correct == test
    test = cartesian_product({"s": ["a", "b"], "i": [5, 6]})
    correct = [
        (["s@a", "i@5"], {"s": "a", "i": 5}),
        (["s@a", "i@6"], {"s": "a", "i": 6}),
        (["s@b", "i@5"], {"s": "b", "i": 5}),
        (["s@b", "i@6"], {"s": "b", "i": 6}),
    ]
    assert correct == test


def test_affine_projection_matrix():
    """
    Test function that creates affine projection matrix.
    """
    ground_truth = torch.Tensor(
        [[2 / 3, -1 / 3, -1 / 3], [-1 / 3, 2 / 3, -1 / 3], [-1 / 3, -1 / 3, 2 / 3]]
    )
    func_output = affine_projection_matrix(3)

    assert torch.allclose(ground_truth, func_output)


def test_project_betas():
    """
    Test function for beta projection.
    """
    # Say we have a vector with 50 elements
    # Test this 10 times:
    for i in range(10):
        test_beta_vec = (-12 - 2) * torch.rand(50) + 2
        new_betas = project_betas(test_beta_vec)
        # Correct vector dimensions
        assert new_betas.shape[0] == 50
        assert len(new_betas.shape) == 1
        # Average of -1 as desired
        assert round(torch.mean(new_betas).item(), 5) == -1
