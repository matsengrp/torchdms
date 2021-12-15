"""Evaluating models."""

import pandas as pd
from torchdms.data import SplitDataset
from torchdms.utils import positions_in_list

QUALITY_CUTOFFS = [-3.0, -1.0]


def build_evaluation_dict(model, test_data, device="cpu"):
    """Evaluate & organize all testing data paired with metadata.

    A function which takes a trained model, matching test
    dataset (BinaryMapDataset w/ the same input dimensions.)
    and return a dictionary containing the

    - samples: binary encodings numpy array shape (num samples, num possible mutations)
    - predictions and targets: both numpy arrays of shape (num samples, num targets)

    This should have everything mostly needed to do plotting
    about testing data (not things like loss or latent space prediction).
    """

    assert test_data.feature_count() == model.input_size
    assert test_data.target_count() == model.output_size
    model.eval()
    samples = test_data.samples.detach().numpy()
    predictions = model(test_data.samples.to(device)).detach().numpy()
    targets = test_data.targets.detach().numpy()

    return {
        "samples": samples,
        "predictions": predictions,
        "targets": targets,
        "original_df": test_data.original_df,
        "wtseq": test_data.wtseq,
        "target_names": test_data.target_names,
    }


def error_df_of_evaluation_dict(evaluation_dict):
    """Build a dataframe that describes the error per test point."""

    def error_df_of_target_idx(target_idx):
        assert target_idx < len(evaluation_dict["target_names"])
        return pd.DataFrame(
            {
                "observed": evaluation_dict["targets"][:, target_idx],
                "predicted": evaluation_dict["predictions"][:, target_idx],
                "n_aa_substitutions": evaluation_dict["original_df"][
                    "n_aa_substitutions"
                ],
                "target": evaluation_dict["target_names"][target_idx],
            }
        )

    error_df = pd.concat(
        map(error_df_of_target_idx, range(len(evaluation_dict["target_names"])))
    )
    error_df["abs_error"] = (error_df["observed"] - error_df["predicted"]).abs()

    error_df["observed_quality"] = positions_in_list(
        error_df["observed"], QUALITY_CUTOFFS
    )

    return error_df


def error_summary_of_error_df(error_df, model):
    """Build a dataframe summarizing error."""
    error_summary_groupby = error_df.groupby(
        ["observed_quality", "target", "n_aa_substitutions"]
    )
    error_summary_df = error_summary_groupby.mean()
    error_summary_df["model"] = model.str_summary()
    return error_summary_df


def error_summary_of_data(data, model, split_label=None, **kwargs):
    error_df = error_df_of_evaluation_dict(build_evaluation_dict(model, data, **kwargs))
    error_summary_df = error_summary_of_error_df(error_df, model)
    if split_label is not None:
        error_summary_df["split_label"] = split_label
    return error_summary_df


def complete_error_summary(data: SplitDataset, model, **kwargs):
    return pd.concat(
        [
            error_summary_of_data(data, model, split_label, **kwargs)
            for split_label, data in data.labeled_splits
        ]
    )
