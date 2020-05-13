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
