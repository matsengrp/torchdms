"""
Testing for utils.py.
"""
import torch
from torchdms.utils import cartesian_product, get_mutation_indicies, parse_epitopes
import torchdms.model


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
    # Epitope dicts for testing (hypothetical 15 site protein)
    epitope_dict = {"1": ["1-3"], "2": ["5-8", "10-12"]}
    # mini alphabet for easy mental calculations.
    alphabet = set(range(5))
    beta_dim = len(alphabet) * 15
    model = torchdms.model.EscapeModel(input_size=beta_dim, target_names=[], alphabet=alphabet, num_epitopes=2)
    epitope_one_ground_truth = torch.ones(beta_dim, dtype=torch.bool)
    epitope_two_ground_truth = torch.ones(beta_dim, dtype=torch.bool)
    # Epitope one will have zeros from idx 0:14
    epitope_one_ground_truth[0:14] = 0
    # Epitope two will have zeros from 20:39, 45:59
    epitope_two_ground_truth[20:39] = 0
    epitope_two_ground_truth[45:59] = 0

    epitope_mask = torch.cat((epitope_one_ground_truth, epitope_two_ground_truth)).reshape(beta_dim, 2)

    parse_epitopes_output = parse_epitopes(epitope_dict, model)
    torch.equal(epitope_mask, parse_epitopes_output)
