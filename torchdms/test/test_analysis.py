"""
Testing for helper methods in analysis.py
"""
import numpy as np
import torch
import torch.nn as nn
import os
import copy
import pkg_resources
from pytest import approx
from torchdms.analysis import Analysis
from torchdms.analysis import _low_rank_approximation
from torchdms.utils import (
    from_pickle_file,
    parse_sites,
    make_all_possible_mutations,
    get_observed_training_mutations,
)
from torchdms.loss import l1
import torchdms.model
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
aux_path = model_path + "_details.pkl"
escape_aux_path = escape_model_path + "_details.pkl"


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
        strata_ceiling=None,
        export_dataframe=None,
        partition_label=None,
    )
    escape_split_df = partition(
        escape_data,
        per_stratum_variants_for_test=10,
        skip_stratum_if_count_is_smaller_than=30,
        strata_ceiling=None,
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
    model = torchdms.model.FullyConnected(
        [1, 10],
        [None, nn.ReLU()],
        split_df_prepped.test.feature_count(),
        split_df_prepped.test.target_names,
        split_df_prepped.test.alphabet,
    )

    # Escape models
    escape_model = torchdms.model.Escape(
        2,
        escape_split_df_prepped.test.feature_count(),
        escape_split_df_prepped.test.target_names,
        escape_split_df_prepped.test.alphabet,
    )

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
    os.remove(aux_path)
    os.remove(escape_aux_path)
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
    approx_est = _low_rank_approximation(test_matrix, 1)
    # assert that values match up
    assert torch.allclose(torch.from_numpy(approx_true), approx_est, rtol=0.001)


def test_project_betas():
    """Test to ensure we get an average value of -1 for non-WT betas."""
    for latent_dim in range(analysis.model.latent_dim):
        assert torch.mean(
            analysis.model.beta_coefficients()[latent_dim, analysis.mutant_idxs]
        ).item() == approx(-1)


def test_zeroed_wt_betas():
    """Test to ensure WT betas of a model are (and remain) 0."""
    for analysis_ in (analysis, escape_analysis):
        wt_idxs = analysis_.val_data.wt_idxs
        # Assert that wt betas are 0 upon initializaiton.
        for latent_dim in range(analysis_.model.latent_dim):
            for idx in wt_idxs:
                assert analysis_.model.beta_coefficients()[latent_dim, int(idx)] == 0

        # Train model with analysis object for 1 epoch
        analysis_.train(**training_params)

        # Assert that wt betas are still 0.
        for latent_dim in range(analysis_.model.latent_dim):
            for idx in wt_idxs:
                assert analysis_.model.beta_coefficients()[latent_dim, int(idx)] == 0


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


def test_seq_to_binary():
    """
    Test function for translating strings of aa-seqs to their 1-hot encoding.
    """
    # The test_seq_to_binary() method will take a string of amino acids and a model.
    wtseq = analysis.val_data.wtseq
    mutseq_valid = "NHT"
    assert wtseq == "NIT"

    # Ground truth indicies for valid cases
    wt_ground_truth = torch.zeros(
        len(wtseq) * len(analysis.model.alphabet), dtype=torch.float
    )
    mut_ground_truth = torch.zeros(
        len(wtseq) * len(analysis.model.alphabet), dtype=torch.float
    )
    wt_ground_truth[analysis.wt_idxs] = 1
    mut_ground_truth[torch.tensor([11, 27, 58], dtype=torch.long)] = 1
    wt_test = analysis.model.seq_to_binary(wtseq)
    mut_test = analysis.model.seq_to_binary(mutseq_valid)

    # Make sure translation works.
    assert torch.equal(wt_ground_truth, wt_test)
    assert torch.equal(mut_ground_truth, mut_test)


def test_concentrations_stored():
    """Tests to make sure Escape() is recieving concentration values as planned (tacking values on to end of encoding)."""
    # Make sure the model's input size doesn't change
    assert escape_model.input_size == len(escape_model.alphabet) * len(
        escape_analysis.val_data.wtseq
    )
    # Ensure that concentrations are in dataframe
    assert "concentration" in escape_analysis.val_data.original_df.columns
    # and the concentrations attribute is true...
    assert escape_analysis.val_data.samples_concentrations is not None


def test_escape_concentrations_forward():
    """Test to make sure concentrations aren't influencing betas in Escape."""
    # Make sure beta_coefficients() only returns sequence indices.
    # The encoding for the polyclonal escape simulated data is 4221 slots.
    # We have 2 sites in the test model.
    test_dict = {"1": ["1-5"], "2": ["10-15"]}
    all_indicies = np.arange(escape_model.input_size)
    site_indicies = parse_sites(test_dict, escape_model)

    assert escape_analysis.model.beta_coefficients().shape == (2, 4221)

    escape_model.randomize_parameters()

    # Jumble betas
    not torch.allclose(
        escape_model.beta_coefficients(),
        torch.zeros_like(escape_model.beta_coefficients()),
    )

    # Now mask them.
    escape_model.fix_gauge(escape_analysis.gauge_mask)

    # Loop through sites.
    for site_id, sites in test_dict.items():
        latent_dim = int(site_id) - 1
        zero_beta_indicies = torch.from_numpy(
            np.setxor1d(all_indicies, site_indicies[latent_dim].numpy())
        )
        non_site_betas = escape_model.beta_coefficients()[
            latent_dim, zero_beta_indicies
        ]
        site_betas = escape_model.beta_coefficients()[
            latent_dim, site_indicies[latent_dim]
        ]
        # Check that all non-site betas are zero.
        torch.allclose(non_site_betas, torch.zeros_like(non_site_betas))
        # Check that the correct site was preserved.
        not torch.allclose(site_betas, torch.zeros_like(site_betas))


def test_stored_unseen_mutations():
    """Test to make sure the set of unseen mutations are stored properly."""
    all_possible_muts = make_all_possible_mutations(
        analysis.val_data.wtseq, analysis.val_data.alphabet
    )
    observed_muts = get_observed_training_mutations(analysis.train_datasets)
    unseen_muts = all_possible_muts.difference(observed_muts)
    assert analysis.unseen_mutations == unseen_muts


def test_latent_origin():
    """The WT sequence should lie at the origin of the latent space in all models."""
    for analysis_ in (analysis, escape_analysis):

        wt_rep = torch.unsqueeze(
            analysis_.model.seq_to_binary(analysis_.val_data.wtseq), 0
        )

        # check on init
        z_WT = analysis_.model.to_latent(wt_rep)
        assert torch.equal(z_WT, torch.zeros_like(z_WT))

        # Train model with analysis object for 1 epoch
        analysis_.train(**training_params)

        # check after training
        z_WT = analysis_.model.to_latent(wt_rep)
        assert torch.equal(z_WT, torch.zeros_like(z_WT))
