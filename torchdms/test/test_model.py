"""
Testing for methods in model.py
"""
import numpy as np
import torch
import os
import pkg_resources
from pytest import approx
from torchdms.analysis import Analysis
from torchdms.analysis import low_rank_approximation
from torchdms.utils import (
    from_pickle_file,
    parse_epitopes,
)
from torchdms.loss import l1
from torchdms.model import model_of_string
from torchdms.data import partition, prep_by_stratum_and_export

TEST_DATA_PATH = pkg_resources.resource_filename("torchdms", "data/test_df.pkl")
ESCAPE_TEST_DATA_PATH = pkg_resources.resource_filename(
    "torchdms", "data/test_escape_df.pkl"
)
out_path = "test_df-prepped"
escape_out_path = "test_df_escape-prepped"
data_path = out_path + ".pkl"
escape_data_path = escape_out_path + ".pkl"
model_path = "run.model"
escape_model_path = "run.escape.model"


def setup_module(module):
    """Loads in test data and model for future tests."""
    print("NOTE: Setting up testing environment...")
    global model, escape_model, escape_analysis, analysis, training_params
    data, wtseq = from_pickle_file(TEST_DATA_PATH)
    escape_data, escape_wtseq = from_pickle_file(ESCAPE_TEST_DATA_PATH)
    split_df = partition(
        data,
        per_stratum_variants_for_test=10,
        skip_stratum_if_count_is_smaller_than=30,
        export_dataframe=None,
        partition_label=None,
    )
    escape_split_df = partition(
        escape_data,
        per_stratum_variants_for_test=10,
        skip_stratum_if_count_is_smaller_than=30,
        export_dataframe=None,
        partition_label=None,
    )
    prep_by_stratum_and_export(split_df, wtseq, ["affinity_score"], out_path, "", None)
    prep_by_stratum_and_export(
        escape_split_df, escape_wtseq, ["prob_escape"], escape_out_path, "", None
    )

    split_df_prepped = from_pickle_file(data_path)
    escape_split_df_prepped = from_pickle_file(escape_data_path)

    # GE models
    model_string = "FullyConnected(1,identity,10,relu)"
    model = model_of_string(model_string, data_path)

    # Escape models
    escape_model_string = "Escape(2)"
    escape_model = model_of_string(escape_model_string, escape_data_path)

    torch.save(model, model_path)
    torch.save(escape_model, escape_model_path)
    analysis_params = {
        "model": model,
        "model_path": model_path,
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
    escape_analysis = Analysis(**escape_analysis_params)
    print("NOTE: Testing environment setup...")


def teardown_module(module):
    """Tears down testing setup -- removes temp models and data."""
    print("NOTE: Tearing down testing environment...")
    os.remove(model_path)
    os.remove(data_path)
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


def test_epitope_mask():
    """Ensure that a provided EscapeModel epitope mask works as expected. """
    # When epitope masking, a few things to keep in mind:
    # 1. The procedure is performed during EscapeModel's fix_gauge() call.
    # 2. The easiest way to pass in an epitope mask is with a dictionary of sites.
    # 3. The number of epitopes provided must match the number of epitopes in model_string.
    # 4. For each epitope provided, make sure sites not-included are set to zero.
    epitope_dict = {'1': ['1-10'], '2': ['50-60', '70-80']}
    all_indicies = np.arange(0, escape_model.input_size)

    epitope_one = parse_epitopes(epitope_dict, escape_model.alphabet)[0].type(torch.LongTensor)
    epi_one_betas = torch.index_select(escape_model.beta_coefficients()[0], 0, epitope_one)
    # Assign random values to betas (or just set them all to something like 10)
    # Assert that the below test is false (epitopes have not been masked)
    # Call fix_gauge()
    # Assert that the test now passes
    # escape_model.fix_gauge(escape_analysis.gauge_mask, parse_epitopes(epitope_dict, escape_model.alphabet))

    for i in range(escape_model.num_epitopes):
        # Grab the epitope indicies
        epitope = parse_epitopes(epitope_dict, escape_model.alphabet)[i].type(torch.LongTensor)
        # Get indicies of betas that should be zero for this epitope.
        zero_betas = torch.from_numpy(np.setxor1d(epitope.numpy(), all_indicies))
        # Make sure that these betas are actually 0.
        escape_model.beta_coefficients()[i] = torch.FloatTensor(1, escape_model.input_size).uniform_(1,5)
        non_epitope_betas = torch.index_select(escape_model.beta_coefficients()[i], 0, zero_betas)
        torch.allclose(non_epitope_betas.type(torch.LongTensor), torch.ones(len(zero_betas)).type(torch.LongTensor))
