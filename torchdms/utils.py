"""Utility functions."""

from copy import deepcopy
import os
import os.path
import pickle
import json
import re
import pandas as pd
import numpy as np
import dms_variants as dms


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
    with open(path, "r") as file:
        return json.load(file)


def to_json_file(obj, path):
    """Write an object to a JSON file."""
    with open(path, "w") as file:
        json.dump(obj, file, indent=4, sort_keys=True)
        file.write("\n")


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


def build_beta_map(test_data, beta_vec):
    """This function creates a beta matrix for one latent layer of a torchdms
    model.

    Takes a binary map object and beta vector as input. Returns a 21xL
    matrix of beta-coefficients and the amino acid alphabet.
    """

    bmap = dms.binarymap.BinaryMap(
        test_data.original_df,
    )

    wtmask = np.full([len(bmap.alphabet), len(test_data.wtseq)], False, dtype=bool)
    alphabet = bmap.alphabet

    for column_position, aa in enumerate(test_data.wtseq):
        row_position = alphabet.index(aa)
        wtmask[row_position, column_position] = True
    # See model.numpy_single_mutant_predictions for why this transpose is here.
    return (
        beta_vec.reshape(len(test_data.wtseq), len(bmap.alphabet)).transpose(),
        alphabet,
    )
