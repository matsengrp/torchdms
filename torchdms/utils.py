from copy import deepcopy
import os.path
import pickle
import dms_variants as dms
import numpy as np
import pandas as pd
import torchdms.model


def from_pickle_file(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def to_pickle_file(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def monotonic_params_from_latent_space(model: torchdms.model.DMSFeedForwardModel):
    """
        following the hueristic that the input layer of a network
        is named 'input_layer' and the weight bias are denoted:

        layer_name.weight
        layer_name.bias.

        this function returns all the parameters
        to be floored to zero in a monotonic model.
        this is every parameter after the latent space
        excluding bias parameters.
        """
    for name, param in model.named_parameters():
        parse_name = name.split(".")
        is_input_layer = parse_name[0] == "input_layer"
        is_bias = parse_name[1] == "bias"
        if not is_input_layer and not is_bias:
            yield param


def evaluation_dict(model, test_data, device="cpu"):
    """
    Evaluate & Organize all testing data paried with metadata.

    A function which takes a trained model, matching test
    dataset (BinaryMapDataset w/ the same input dimensions.)
    and return a dictionary containing the

    - samples: binary encodings numpy array shape (num samples, num possible mutations)
    - predictions and targets: both numpy arrays of shape (num samples, num targets)
    -

    This should have everything mostly needed to do plotting
    about testing data (not things like loss or latent space prediction)
    """
    # TODO check the testing dataset matches the
    # model input size and output size!

    return {
        "samples": test_data.samples.detach().numpy(),
        "predictions": model(test_data.samples.to(device)).detach().numpy(),
        "targets": test_data.targets.detach().numpy(),
        "original_df": test_data.original_df,
        "wtseq": test_data.wtseq,
        "target_names": test_data.target_names,
    }


def get_first_key_with_an_option(option_dict):
    """
    Return the first key that maps to a list.

    We will call such a key-value pair an "option". An "option dict" will be a dict that
    (may) have such a key-value pair.
    """
    for key, value in option_dict.items():
        if isinstance(value, list):
            return key
    return None


def cartesian_product(option_dict):
    """
    Expand an option dict, collecting the choices made in the first return value of the tuple.

    The best way to understand this function is to look at the test in `test/test_utils.py`.
    """
    return _cartesian_product_aux([([], option_dict)])


def _cartesian_product_aux(list_of_choice_list_and_option_dict_pairs):
    """
    Recursive procedure to assist cartesian_product.
    """
    expanded_something = False
    expanded_list = []
    for choice_list, option_dict in list_of_choice_list_and_option_dict_pairs:
        option_key = get_first_key_with_an_option(option_dict)
        if option_key is None:
            continue
        # else:
        expanded_something = True
        for option_value in option_dict[option_key]:
            key_value_str = str(option_key) + "@" + str(option_value)
            new_option_dict = deepcopy(option_dict)
            new_option_dict[option_key] = option_value
            expanded_list.append((choice_list + [key_value_str], new_option_dict))

    if expanded_something:
        return _cartesian_product_aux(expanded_list)
    # else:
    return list_of_choice_list_and_option_dict_pairs
