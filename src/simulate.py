import collections
import itertools
import random
import tempfile
import time
import warnings

import Bio.Seq

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

from plotnine import *

import dms_variants.binarymap
import dms_variants.codonvarianttable
import dms_variants.simulate
import dms_variants.utils
from dms_variants.constants import CBPALETTE, CODONS_NOSTOP

seed = 1  # random number seed
genelength = 300  # gene length in codons
libs = ['lib_1']  # distinct libraries of gene
variants_per_lib = 500 * genelength  # variants per library
avgmuts = 2.5  # average codon mutations per variant
bclen = 16  # length of nucleotide barcode for each variant
variant_error_rate = 0.005  # rate at which variant sequence mis-called
avgdepth_per_variant = 200  # average per-variant sequencing depth
lib_uniformity = 5  # uniformity of library pre-selection
noise = 0.02  # random noise in selections
bottleneck = 10  # bottleneck from pre- to post-selection, average number of each variant that gets through

# Set seed for reproducibility
random.seed(seed)


# Simulate wildtype sequence
geneseq = ''.join(random.choices(CODONS_NOSTOP, k=genelength))
print(f"Wildtype gene of {genelength} codons:\n \n{geneseq}")

# Wildtype protein sequence
protseq = dms_variants.utils.translate(geneseq)
print(f"Wildtype protein sequence:\n \n{protseq}")


# Create codon variant table
variants = dms_variants.simulate.simulate_CodonVariantTable(
    geneseq=geneseq,
    bclen=bclen,
    library_specs={lib: {'avgmuts': avgmuts,
                         'nvariants': variants_per_lib}
                   for lib in libs},
    seed=seed
)

# Print a few rows from the table
print(variants.barcode_variant_df.head(n=5))

# Simulate phenotype
phenosimulator = dms_variants.simulate.SigmoidPhenotypeSimulator(
    geneseq, seed=seed)

# Simulate variant counts
counts = dms_variants.simulate.simulateSampleCounts(
    variants=variants,
    phenotype_func=phenosimulator.observedEnrichment,
    variant_error_rate=variant_error_rate,
    pre_sample={'total_count': int(variants_per_lib * avgdepth_per_variant),
                'uniformity': lib_uniformity},
    pre_sample_name='pre-selection',
    post_samples={'post-selection':
                  {'noise': noise,
                   'total_count': int(variants_per_lib * avgdepth_per_variant),
                   'bottleneck': int(bottleneck * variants_per_lib)}
                  },
    seed=seed
)

# View the variants
print(counts.head(n=5))

# Add counts to variants table
variants.add_sample_counts_df(counts)
print(variants.n_variants_df())

# Assign functional scores
func_scores = variants.func_scores('pre-selection', libraries=libs)

# Print functional scores dataframe
print(func_scores.head(n=5).round(3))

# Create Binary Map
bmap = dms_variants.binarymap.BinaryMap(func_scores,
                                        expand=True,
                                        wtseq=protseq)
bmap_df = bmap.binary_variants.toarray()

# Take a look at the variants and corresponding functional scores
print("Binary Variants: ")
print(bmap_df[0:5])
print("\n")
print("Functional Scores: \n")
print(bmap.func_scores)

# Write simulated data to files:
sparse.save_npz("../data/dms_simulation_150000_variants.npz", bmap.binary_variants)
np.savetxt("../data/dms_simulation_150000_variants_funcscores.txt", bmap.func_scores,
           delimiter='\t')
np.savetxt("../data/dms_simulation_150000_variants_funcscores_var.txt", bmap.func_scores_var,
           delimiter='\t')
