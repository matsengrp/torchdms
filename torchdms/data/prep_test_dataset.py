"""
Some code from Frances Welsh, useful for prepping our mini test data set.
"""
import random
import pandas as pd
from torchdms.utils import to_pickle_file


def random_char_from_string(string):
    """
    Returns a randomly chosen character from a string.
    """
    random_loc = random.randint(0, len(string) - 1)
    return string[random_loc]


def generate_wt_dict(wtseq):
    """
    Generates 'wtdict' from a sequence, where keys are the WT amino acid and
    position, and values are legal amino acid substitutions.
    """
    amino_acids = "GPAVLIMCFYWHKRQNEDST"
    wtdict = {}
    i = 1
    for aa in wtseq:
        wtdict[f"{aa}{i}"] = amino_acids.replace(f"{aa}", "")
        i += 1
    return wtdict


def generate_tiny_dataset(df, n_aa_subs=3, n_samples=150, wtseq="NIT"):
    """
    Generates a small dataset for use in testing, and exports as .pkl file with
    small wtseq.

    Samples 'n_samples' number of rows from the inputted dataset, with equal
    proportions from variants with between 1 and 'n_aa_subs' number of
    mutations. The 'aa_substitutions' column is populated with randomly
    generated substitutions within the length of wtseq. Note that n_aa_subs
    should not exceed the length of wtseq, and n_samples should be divisible by
    n_aa_subs.

    Duplicate substitutions will be present if wtseq does not exceed 20 amino
    acids, as there will by definition be duplicate single aa substitutions for
    each wt position.
    """

    wtdict = generate_wt_dict(wtseq)

    sample_dfs = []
    n_samples_per_sub = int(n_samples / n_aa_subs)
    for n_substutions in range(1, n_aa_subs + 1):
        n_mut_df = df[df["n_aa_substitutions"] == n_substutions].sample(
            n=n_samples_per_sub
        )

        for index, _ in n_mut_df.iterrows():
            prefixes = random.sample(list(wtdict), n_substutions)
            aa_subs = []
            for prefix in prefixes:
                single_aa_sub = prefix + (random_char_from_string(wtdict[prefix]))
                aa_subs.append(single_aa_sub)
            full_aa_sub = " ".join(aa_subs)

            n_mut_df.at[index, "aa_substitutions"] = full_aa_sub

        sample_dfs.append(n_mut_df)

    test_df = pd.concat(sample_dfs)
    to_pickle_file([test_df, wtseq], "test_df.pkl")
