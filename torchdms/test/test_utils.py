"""
Testing for utils.py.
"""
import torch
from torchdms.utils import (
    cartesian_product,
    affine_projection_matrix,
    get_mutation_indicies,
)


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


def test_get_mutation_indicies():
    """
    Test function to get mutation indicies.
    """
    alphabet = {"A": 0, "B": 1, "C": 2}
    mutations = ["A1B", "C2A", "B3C"]
    ground_truth = torch.Tensor([1, 3, 8]).type(torch.long)
    unseen_muts = get_mutation_indicies(mutations, alphabet)

    assert torch.allclose(ground_truth, unseen_muts)
