"""
Testing for helper methods in analysis.py
"""
import numpy as np
import torch
import os
import pkg_resources
from pytest import approx
from torchdms.analysis import Analysis
from torchdms.analysis import low_rank_approximation
from torchdms.utils import from_pickle_file, parse_epitopes
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
    escape_training_params = {"epoch_count": 1, "loss_fn": l1}
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


def test_low_rank_approximation():
    """Tests low-rank approximation function."""
    # define simple 2-rank matrix
    test_matrix = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]], dtype="float")

    # store true 1-rank approximation here & flatten column-wise
    approx_true = np.array(
        [
            [1.736218, 4.207153, 6.678088],
            [2.071742, 5.020186, 7.968631],
            [2.407267, 5.833220, 9.259173],
        ]
    ).flatten("F")

    # take low-rank (1) approximation
    approx_est = low_rank_approximation(test_matrix, 1)
    # assert that values match up
    assert torch.allclose(torch.from_numpy(approx_true), approx_est, rtol=0.001)


def test_concentrations_stored():
    """Tests to make sure EscapeModel() is recieving concentration values as planned (tacking values on to end of encoding)."""
    # Make sure the model's input size doesn't change
    assert escape_model.input_size == len(escape_model.alphabet) * len(
        escape_analysis.val_data.wtseq
    )
    # Ensure that concentrations are in dataframe
    assert "concentration" in escape_analysis.val_data.original_df.columns
    # and the concentrations attribute is true...
    assert escape_analysis.val_data.samples_concentrations is not None


def test_escape_concentrations_forward():
    """Test to make sure concentrations aren't influencing betas in EscapeModel."""
    # Make sure beta_coefficients() only returns sequence indices.
    # The encoding for the polyclonal escape simulated data is 4221 slots.
    # We have 2 epitopes in the test model.
    test_dict = {"1": ["1-5"], "2": ["10-15"]}
    all_indicies = np.arange(escape_model.input_size)
    epitope_indicies = parse_epitopes(test_dict, escape_model.alphabet)

    assert escape_analysis.model.beta_coefficients().shape == (2, 4221)

    escape_model.randomize_parameters()

    # Jumble betas
    not torch.allclose(
        escape_model.beta_coefficients(),
        torch.zeros_like(escape_model.beta_coefficients()),
    )

    # Now mask them.
    escape_model.fix_gauge(escape_analysis.gauge_mask, epitope_indicies)

    # Loop through epitopes.
    for epitope_id, sites in test_dict.items():
        latent_dim = int(epitope_id) - 1
        zero_beta_indicies = torch.from_numpy(
            np.setxor1d(all_indicies, epitope_indicies[latent_dim].numpy())
        )
        non_epitope_betas = escape_model.beta_coefficients()[
            latent_dim, zero_beta_indicies
        ]
        epitope_betas = escape_model.beta_coefficients()[
            latent_dim, epitope_indicies[latent_dim]
        ]
        # Check that all non-epitope betas are zero.
        torch.allclose(non_epitope_betas, torch.zeros_like(non_epitope_betas))
        # Check that the correct epitope was preserved.
        not torch.allclose(epitope_betas, torch.zeros_like(epitope_betas))
