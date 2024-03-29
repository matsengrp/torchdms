"""
Testing model module
"""

import torch
import numpy as np
import torchdms.model
import torch.nn as nn
import os
import copy
import pkg_resources
from pytest import approx
from torchdms.analysis import Analysis
from torchdms.utils import from_pickle_file
from torchdms.loss import l1
from torchdms.data import partition, prep_by_stratum_and_export


# vanilla model paths
TEST_DATA_PATH = pkg_resources.resource_filename("torchdms", "data/test_df.pkl")
out_path = "test_df-prepped"
data_path = out_path + ".pkl"

model_path = "run.model"
aux_path = model_path + "_details.pkl"

# bias/no bias model paths
bias_model_path = "run.bias.model"
bias_aux_path = bias_model_path + "_details.pkl"

# monotonic model path
monotonic_model_path = "run.monotonic.model"
monotonic_aux_path = monotonic_model_path + "_details.pkl"

# 2D model paths
# TEST_2D_DATA_PATH = pkg_resources.resource_filename("torchdms", "data/test_df.pkl")

# Escape model paths
ESCAPE_TEST_DATA_PATH = pkg_resources.resource_filename(
    "torchdms", "data/test_escape_df.pkl"
)
escape_out_path = "test_df_escape-prepped"
escape_data_path = escape_out_path + ".pkl"
escape_model_path = "run.escape.model"


def setup_module(module):
    """Loads in test data and model for future tests."""
    print("NOTE: Setting up testing environment...")
    global training_params
    global model, analysis
    global escape_model, escape_analysis
    global bias_model, bias_analysis
    global monotonic_model, monotonic_analysis

    data, wtseq = from_pickle_file(TEST_DATA_PATH)
    split_df = partition(
        data,
        per_stratum_variants_for_test=10,
        skip_stratum_if_count_is_smaller_than=30,
        strata_ceiling=None,
        export_dataframe=None,
        partition_label=None,
    )
    prep_by_stratum_and_export(split_df, wtseq, ["affinity_score"], out_path, "", None)
    split_df_prepped = from_pickle_file(data_path)

    escape_data, escape_wtseq = from_pickle_file(ESCAPE_TEST_DATA_PATH)
    escape_split_df = partition(
        escape_data,
        per_stratum_variants_for_test=10,
        skip_stratum_if_count_is_smaller_than=30,
        strata_ceiling=None,
        export_dataframe=None,
        partition_label=None,
    )
    prep_by_stratum_and_export(
        escape_split_df, escape_wtseq, ["prob_escape"], escape_out_path, "", None
    )

    escape_split_df_prepped = from_pickle_file(escape_data_path)

    # GE models (w/o non-lin and output bias parameters)
    model = torchdms.model.FullyConnected(
        [1, 10],
        [None, nn.ReLU()],
        split_df_prepped.test.feature_count(),
        split_df_prepped.test.target_names,
        split_df_prepped.test.alphabet,
    )

    # GE models (w/ non-lin and output bias parameters)
    bias_model = torchdms.model.FullyConnected(
        [1, 10],
        [None, nn.ReLU()],
        split_df_prepped.test.feature_count(),
        split_df_prepped.test.target_names,
        split_df_prepped.test.alphabet,
        non_lin_bias=True,
        output_bias=True,
    )

    # monotonically increasing model
    monotonic_model = torchdms.model.FullyConnected(
        [1, 10],
        [None, nn.ReLU()],
        split_df_prepped.test.feature_count(),
        split_df_prepped.test.target_names,
        split_df_prepped.test.alphabet,
        # non_lin_bias=True,
        # output_bias=True,
        monotonic_sign=1,
    )

    # Escape models
    escape_model = torchdms.model.Escape(
        2,
        escape_split_df_prepped.test.feature_count(),
        escape_split_df_prepped.test.target_names,
        escape_split_df_prepped.test.alphabet,
    )

    torch.save(model, model_path)
    torch.save(bias_model, bias_model_path)
    torch.save(monotonic_model, monotonic_model_path)
    torch.save(escape_model, escape_model_path)

    analysis_params = {
        "model": model,
        "model_path": model_path,
        "val_data": split_df_prepped.val,
        "train_data_list": split_df_prepped.train,
    }

    bias_analysis_params = {
        "model": bias_model,
        "model_path": bias_model_path,
        "val_data": split_df_prepped.val,
        "train_data_list": split_df_prepped.train,
    }

    monotonic_analysis_params = {
        "model": monotonic_model,
        "model_path": monotonic_model_path,
        "val_data": split_df_prepped.val,
        "train_data_list": split_df_prepped.train,
    }

    escape_analysis_params = {
        "model": escape_model,
        "model_path": escape_model_path,
        "val_data": escape_split_df_prepped.val,
        "train_data_list": escape_split_df_prepped.train,
    }

    training_params = {"epoch_count": 1, "loss_fn": l1}
    analysis = Analysis(**analysis_params)
    bias_analysis = Analysis(**bias_analysis_params)
    monotonic_analysis = Analysis(**monotonic_analysis_params)
    escape_analysis = Analysis(**escape_analysis_params)
    print("NOTE: Testing environment setup...")


def teardown_module(module):
    """Tears down testing setup -- removes temp models and data."""
    print("NOTE: Tearing down testing environment...")
    os.remove(data_path)

    os.remove(model_path)
    os.remove(aux_path)

    os.remove(bias_model_path)
    os.remove(bias_aux_path)

    os.remove(monotonic_model_path)
    os.remove(monotonic_aux_path)

    os.remove(escape_data_path)
    os.remove(escape_model_path)

    print("NOTE: Testing environment torn down...")


# GAUGE FIXING TESTS
def test_project_betas():
    """Test to ensure we get an average value of -1 for non-WT betas."""
    for latent_dim in range(analysis.model.latent_dim):
        assert torch.mean(
            analysis.model.beta_coefficients()[latent_dim, analysis.mutant_idxs]
        ).item() == approx(-1)


def test_zeroed_wt_betas():
    """Test to ensure WT betas of a model are (and remain) 0."""
    wt_idxs = analysis.val_data.wt_idxs
    assert analysis.val_data.wtseq == "NIT"
    # Assert that wt betas are 0 upon initializaiton.
    for latent_dim in range(analysis.model.latent_dim):
        for idx in wt_idxs:
            assert analysis.model.beta_coefficients()[latent_dim, int(idx)] == 0

    # Train model with analysis object for 1 epoch
    analysis.train(**training_params)

    # Assert that wt betas are still 0.
    for latent_dim in range(analysis.model.latent_dim):
        for idx in wt_idxs:
            assert analysis.model.beta_coefficients()[latent_dim, int(idx)] == 0


def test_zeroed_unseen_betas():
    """Test to ensure unseen betas of a model are (and remain) 0."""
    unseen_idxs = analysis.unseen_idxs
    assert analysis.val_data.wtseq == "NIT"
    # Assert that wt betas are 0 upon initializaiton.
    for latent_dim in range(analysis.model.latent_dim):
        for idx in unseen_idxs:
            assert analysis.model.beta_coefficients()[latent_dim, int(idx)] == 0

    # Train model with analysis object for 1 epoch
    analysis.train(**training_params)

    # Assert that wt betas are still 0.
    for latent_dim in range(analysis.model.latent_dim):
        for idx in unseen_idxs:
            assert analysis.model.beta_coefficients()[latent_dim, int(idx)] == 0


def test_bias_contrait_predicts_zero_on_wt():
    """test to ensure the bias constrained (default) model predicts zero on wt seq"""
    unseen_idxs = analysis.unseen_idxs
    assert analysis.val_data.wtseq == "NIT"
    # Assert that wt betas are 0 upon initializaiton.
    for latent_dim in range(analysis.model.latent_dim):
        for idx in unseen_idxs:
            assert analysis.model.beta_coefficients()[latent_dim, int(idx)] == 0

    # Train model with analysis object for 1 epoch
    analysis.train(**training_params)

    wt_pred = analysis.model(analysis.model.seq_to_binary(analysis.val_data.wtseq))
    assert torch.equal(wt_pred, torch.tensor([0.0]))


def test_bias_predicts_non_zero_on_wt():
    """test to ensure the bias model predicts non-zero on wt seq"""
    unseen_idxs = bias_analysis.unseen_idxs
    assert bias_analysis.val_data.wtseq == "NIT"
    # Assert that wt betas are 0 upon initializaiton.
    for latent_dim in range(bias_analysis.model.latent_dim):
        for idx in unseen_idxs:
            assert bias_analysis.model.beta_coefficients()[latent_dim, int(idx)] == 0

    # Train model with bias_analysis object for 1 epoch
    bias_analysis.train(**training_params)

    wt_pred = bias_analysis.model(
        bias_analysis.model.seq_to_binary(bias_analysis.val_data.wtseq)
    )
    assert not torch.equal(wt_pred, torch.tensor([0.0]))


def test_monotonic_params_from_latent_space():
    """test to make sure that we're getting the correct parameters
    from the method that returns all monotonic params"""

    for param in monotonic_analysis.model.monotonic_params_from_latent_space():
        assert np.all(torch.gt(param.data, 0).detach().numpy())

    # Train model with bias_analysis object for 1 epoch
    monotonic_analysis.train(**training_params)
