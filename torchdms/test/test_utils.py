"""
Testing for utils.py.
"""
import torch
from torchdms.utils import cartesian_product, get_mutation_indicies, parse_epitopes


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


def test_get_mutation_indicies():
    """
    Test function to get mutation indicies.
    """
    alphabet = {"A": 0, "B": 1, "C": 2}
    mutations = ["A1B", "C2A", "B3C"]
    ground_truth = torch.Tensor([1, 3, 8]).type(torch.long)
    unseen_muts = get_mutation_indicies(mutations, alphabet)

    assert torch.allclose(ground_truth, unseen_muts)


def test_parse_epitopes():
    """
    Ensure linear and conformational epitopes are read properly.
    """
    # Epitope dicts for testing (hypothetical 10 site protein)
    epitope_dict = {"1": ["1-3"], "2": ["5-8", "10-12"]}
    alphabet = set(range(5))
    epitope_one = torch.arange(0, 15).type(torch.LongTensor)
    epitope_two = torch.cat((torch.arange(20, 40), torch.arange(45, 60))).type(
        torch.LongTensor
    )

    # Parse epitope dicts to get beta indicies.
    linear_idxs = parse_epitopes(epitope_dict, alphabet)

    # Linear and conformational epitope tests
    assert torch.allclose(linear_idxs[0], epitope_one)
    assert torch.allclose(linear_idxs[1], epitope_two)
