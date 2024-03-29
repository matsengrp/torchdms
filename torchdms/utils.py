"""Utility functions."""

from copy import deepcopy
import os
import os.path
import pickle
import json
import re
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional


def from_pickle_file(path):
    """Load an object from a pickle file."""
    with open(path, "rb") as file:
        return pickle.load(file)


def to_pickle_file(obj, path):
    """Write an object to a pickle file."""
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def from_json_file(path):
    """Load an object from a JSON file."""
    with open(path, "r", encoding="UTF-8") as file:
        return json.load(file)


def to_json_file(obj, path):
    """Write an object to a JSON file."""
    with open(path, "w", encoding="UTF-8") as file:
        json.dump(obj, file, indent=4, sort_keys=True)
        file.write("\n")


def float_list_of_comma_separated_string(in_str):
    """Parse a string into a list of floats."""
    if in_str is None:
        return None
    # else
    return [float(x) for x in in_str.split(",")]


def count_variants_with_a_mutation_towards_an_aa(series_of_aa_substitutions, aa):
    """Count mutations towards a given amino acid.

    If an amino acid appears multiple times at different positions, we
    only count it once.
    """
    count = 0
    for substitution_string in series_of_aa_substitutions:
        for substitution in substitution_string.split():
            if substitution[-1] == aa:
                count += 1
                break
    return count


def make_legal_filename(label):
    """Remove spaces and non-alphanumeric characters from a given string."""
    legal_filename = label.replace(" ", "_")
    legal_filename = "".join(x for x in legal_filename if x.isalnum())
    return legal_filename


def positions_in_list(series, items):
    """Give the positions of the things in a series relative to a list of
    items.

    Example:
    >>> positions_in_list(pd.Series([4., -3., -1., 9.]), [-2., 0.])
    0    2
    1    0
    2    1
    3    2
    dtype: int64

    The first item is 2 because 4 is bigger than -2 and 0.
    The second item is 0 because -3 is smaller than everything.
    The third item is 1 because -1 is between -2 and 0.
    """
    assert items == sorted(items)
    positions = pd.Series(data=0, index=series.index)
    for idx, item in enumerate(items):
        positions[item < series] = idx + 1
    return positions


def get_only_entry_from_constant_list(items):
    """Assert that a list is constant and return that single value."""
    assert len(items) > 0
    first = items[0]
    for item in items[1:]:
        assert first == item
    return first


def get_first_key_with_an_option(option_dict):
    """Return the first key that maps to a list.

    We will call such a key-value pair an "option". An "option dict"
    will be a dict that (may) have such a key-value pair.
    """
    for key, value in option_dict.items():
        if isinstance(value, list):
            return key
    return None


def cat_list_values(list_valued_dict, desired_keys):
    """
    >>> cat_list_values({"a":[1,2], "b":[3,4], "c":[5]}, ["a", "c"])
    [1, 2, 5]
    """
    output = []
    for key in desired_keys:
        output += list_valued_dict[key]
    return output


def cartesian_product(option_dict):
    """Expand an option dict, collecting the choices made in the first return
    value of the tuple.

    The best way to understand this function is to look at the test in
    `test/test_utils.py`.
    """
    return _cartesian_product_aux([([], option_dict)])


def defunkified_str(in_object):
    """Apply str, then replace shell-problematic characters with
    underscores."""
    return re.sub(r"[(),]", "_", str(in_object))


def _cartesian_product_aux(list_of_choice_list_and_option_dict_pairs):
    """Recursive procedure to assist cartesian_product."""
    expanded_something = False
    expanded_list = []
    for choice_list, option_dict in list_of_choice_list_and_option_dict_pairs:
        option_key = get_first_key_with_an_option(option_dict)
        if option_key is None:
            continue
        # else:
        expanded_something = True
        for option_value in option_dict[option_key]:
            key_value_str = str(option_key) + "@" + defunkified_str(option_value)
            new_option_dict = deepcopy(option_dict)
            new_option_dict[option_key] = option_value
            expanded_list.append((choice_list + [key_value_str], new_option_dict))

    if expanded_something:
        return _cartesian_product_aux(expanded_list)
    # else:
    return list_of_choice_list_and_option_dict_pairs


def make_cartesian_product_hierarchy(dict_of_option_dicts, dry_run=False):
    """Make a directory hierarchy, starting with `_output`, expanding the
    option_dict via a cartesian product."""
    for master_key, option_dict in dict_of_option_dicts.items():
        for choice_list, choice_dict in cartesian_product(option_dict):
            final_dict = {master_key: choice_dict}
            assert choice_list
            directory_path = os.path.join("_output", *choice_list)
            json_path = os.path.join(directory_path, "config.json")
            print(json_path)
            if not dry_run:
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                to_json_file(final_dict, json_path)


def build_beta_map(wtseq, alphabet, beta_vec):
    """This function creates a beta matrix for one latent layer of a torchdms
    model.

    Takes a binary map object and beta vector as input. Returns a 21xL
    matrix of beta-coefficients and the amino acid alphabet.
    """

    wtmask = np.full([len(alphabet), len(wtseq)], False, dtype=bool)

    for column_position, aa in enumerate(wtseq):
        row_position = alphabet.index(aa)
        wtmask[row_position, column_position] = True
    # See model.numpy_single_mutant_predictions for why this transpose is here.
    return beta_vec.reshape(len(wtseq), len(alphabet)).transpose()


def make_all_possible_mutations(wtseq, alphabet):
    """This function creates a set of all possible amino acid substitutions.

    Takes a wild type sequence, and a character alphabet. Returns a 20*L
    list of possible mutations.
    """
    all_possible_mutations = [
        wt_aa + str(site + 1) + alt_aa
        for site, wt_aa in enumerate(wtseq)
        for alt_aa in alphabet
        if alt_aa != wt_aa
    ]
    # make sure all mutations from WT are stored.
    assert len(all_possible_mutations) == (len(alphabet) - 1) * len(wtseq)
    assert len(set(all_possible_mutations)) == len(all_possible_mutations)
    return set(all_possible_mutations)


def get_observed_training_mutations(train_data_list):
    """Returns a list of all aa subs in training data list."""
    observed_mutations = set()
    for train_dataset in train_data_list:
        train_muts = train_dataset.original_df["aa_substitutions"]
        train_muts_split = [sub for muts in train_muts for sub in muts.split()]
        observed_mutations.update(train_muts_split)
    return observed_mutations


def get_mutation_indicies(mutation_list, alphabet):
    """Returns a list of beta indicies for a given list of mutations(aa - site -
    aa fomat)."""
    indicies = []
    alphabet_dict = {letter: idx for idx, letter in enumerate(alphabet)}
    for mut in mutation_list:
        mut_aa = mut[-1]
        site = int(mut[1:-1])
        indicies.append(((site - 1) * len(alphabet_dict)) + alphabet_dict[mut_aa])

    # pylint: disable=not-callable
    return torch.tensor(indicies, dtype=torch.long)


def parse_sites(site_dict, model):
    """Parse site dictionary and return beta indicies for given alphabet."""
    # Assume everything will be set to zero, and set site indicies to 'False'
    site_mask = torch.ones_like(model.beta_coefficients(), dtype=torch.bool)
    for region_id, region_sites in enumerate(site_dict.values()):
        for chunk in region_sites:
            site_1, site_2 = [int(x) for x in chunk.split("-")]
            start = (site_1 - 1) * len(model.alphabet)
            end = start + (site_2 - site_1 + 1) * len(model.alphabet)
            site_mask[region_id, start:end] = False

    return site_mask


def activation_of_string(string):
    if string == "identity" or string is None:
        return nn.Identity()
    # else:
    if hasattr(torch, string):
        return getattr(torch, string)
    # else:
    if hasattr(torch.nn.functional, string):
        return getattr(torch.nn.functional, string)

    raise IOError(f"Don't know activation named {string}.")
