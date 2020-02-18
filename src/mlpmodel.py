import random
import time
import warnings

import pandas as pd
import numpy as np

from plotnine import *

import scipy
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

import dms_variants.binarymap
import dms_variants.codonvarianttable
import dms_variants.globalepistasis
import dms_variants.plotnine_themes
import dms_variants.simulate
from dms_variants.constants import CBPALETTE, CODONS_NOSTOP


# Network class
class Net(nn.Module):

    def __init__(self, input_size, hidden1_size):
        """ Initialize the model.

        Args:
            - self (self): Self reference to the network.
            - input_size (int): Number of features in the input data.
            - hidden1_size (int): Number of nodes in layer 1 of the network.

        Returns:
            - void

        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)  # input -> hidden layer
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden1_size, 1)  # hidden -> output layer

    def forward(self, x):
        """ Makes a pass forward through the network.

        Args:
            - self (self): Self reference to the network.
            - x (tensor): The initial input of the model.

        Return:
            - out (tensor): The values after being pushed through the network
        """
        out = self.fc1(x)
        out = self.sigmoid1(out)
        out = self.fc2(out)
        return out


def train_network(model, n_epoch, batch_size, train_data, train_labels,
                  optimizer, criterion, get_train_loss=False):
    """ Function to train network over a number of epochs, w/ given batch size.

    Args:
        - model (Net): Network model class to be trained.
        - n_epoch (int): The number of epochs (passes through training data)
        - batchsize (int): The number of samples to process in a batch.
        - train_data (ndarray): Input training data.
        - train_labels (ndarray): Corresponding labels/targets for traindata.
        - optimizer (torch Optimizer): A PyTorch optimizer for training.
        - criterion (torch Criterion): A PyTorch criterion for training.
        - get_train_loss (boolean): True if training loss by batch is desired
                                    to be returned.
    Returns:
        - model (Net): The trained Network model.
    """
    losses = []
    for epoch in range(n_epoch):
        permutation = np.random.permutation(train_data.shape[0])

        for i in range(0, train_data.shape[0], batch_size):
            optimizer.zero_grad()
            id = permutation[i:i+batch_size]
            batch_X, batch_y = train_data[id], train_labels[id]

            batch_X = torch.from_numpy(batch_X).float()
            batch_y = torch.from_numpy(batch_y).float()

            # Train the model
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)

            # Backprop and SGD step
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

    if get_train_loss:
        return model, losses
    else:
        return model


# Simulation parameters
seed = 1  # random number seed
genelength = 60  # gene length in codons
variants_per_lib = 500 * genelength  # variants per library
avgmuts = 2.5  # average mutations per variant
bclen = 16  # length of nucleotide barcode for each variant
variant_error_rate = 0.005  # rate at which variant sequence mis-called
avgdepth_per_variant = 100  # average per-variant sequencing depth
lib_uniformity = 5  # uniformity of library pre-selection
noise = 0.02  # random noise in selections
bottleneck = 10  # bottleneck from pre- to post-selection as multiple of variants_per_lib

random.seed(seed)
warnings.simplefilter('ignore')


# Simulate sequence
geneseq = ''.join(random.choices(CODONS_NOSTOP, k=genelength))
print(f"Wildtype gene of {genelength} codons:\n{geneseq}")

variants = dms_variants.simulate.simulate_CodonVariantTable(
    geneseq=geneseq,
    bclen=bclen,
    library_specs={'lib_1': {'avgmuts': avgmuts,
                             'nvariants': variants_per_lib},
                   },
    seed=seed,
)

# Simulate counts for each variant - 2 latent phenotypes.
pheno1 = dms_variants.simulate.SigmoidPhenotypeSimulator(
    geneseq,
    seed=seed,
    wt_latent=4,
    stop_effect=-10,
    norm_weights=[(0.4, -1.25, 1.25),
                  (0.6, -5.5, 2.25),
                  ]
)
pheno2 = dms_variants.simulate.SigmoidPhenotypeSimulator(
    geneseq,
    seed=seed,
    wt_latent=3,
    stop_effect=0,
    norm_weights=[(0.7, 0, 0.1),
                  (0.2, -8, 3),
                  (0.1, -1, 2),
                  ]
)

phenosimulator = dms_variants.simulate.MultiLatentSigmoidPhenotypeSimulator([pheno1, pheno2])

# Simulate counts of variants
counts = dms_variants.simulate.simulateSampleCounts(
    variants=variants,
    phenotype_func=phenosimulator.observedEnrichment,
    variant_error_rate=variant_error_rate,
    pre_sample={'total_count': variants_per_lib * avgdepth_per_variant,
                'uniformity': lib_uniformity},
    pre_sample_name='pre-selection',
    post_samples={'post-selection':
                  {'noise': noise,
                   'total_count': variants_per_lib * avgdepth_per_variant,
                   'bottleneck': variants_per_lib * bottleneck}
                  },
    seed=seed,
)

# Add to the codon variant table
variants.add_sample_counts_df(counts)

# Calculate functional functional scores for variants.
func_scores = variants.func_scores('pre-selection',
                                   libraries=variants.libraries)

# Func score types
func_scores = dms_variants.codonvarianttable.CodonVariantTable.classifyVariants(func_scores)


# make sure we're just fitting one library, if multiple you should fit one-by-one
assert func_scores['library'].nunique() == 1

# create BinaryMap of variants
bmap = dms_variants.binarymap.BinaryMap(func_scores)

# Fit the global epistasis models
fits_df = dms_variants.globalepistasis.fit_models(binarymap=bmap,
                                                  likelihood='Gaussian',
                                                  max_latent_phenotypes=2)

print(fits_df.drop('model', axis='columns').round(1))

# Evaluate models
phenotypes_df = pd.concat(
    [tup.model.phenotypes_df.assign(model=tup.description)
     for tup in fits_df.itertuples()],
    sort=False,
    ignore_index=True)

no_epi = phenotypes_df.loc[phenotypes_df["model"] == "no epistasis"]
one_pheno = phenotypes_df.loc[phenotypes_df["model"] == "global epistasis with 1 latent phenotypes"]
two_pheno = phenotypes_df.loc[phenotypes_df["model"] == "global epistasis with 2 latent phenotypes"]

print("MSE of no epistasis model: {:.3f}".format(
    mean_squared_error(no_epi['func_score'], no_epi['observed_phenotype'])))
print("MSE of global epistasis w/ 1 Latent Phenotype model: {:.3f}".format(
    mean_squared_error(one_pheno['func_score'], one_pheno['observed_phenotype'])))
print("MSE of global epistasis w/ 2 Latent Phenotypes model: {:.3f}".format(
    mean_squared_error(two_pheno['func_score'], two_pheno['observed_phenotype'])))


# Create input data for PyTorch
bmap_torch = torch.from_numpy(bmap.binary_variants.toarray()).float()
func_scores_torch = torch.from_numpy(func_scores['func_score'].to_numpy()).float()
bmap_np = bmap.binary_variants.toarray()
func_scores_np = func_scores['func_score'].to_numpy()

# Instantiate network.
net = Net(input_size=bmap_np.shape[1], hidden1_size=2)
print(net)

# Train network
net.train()
criterion = torch.nn.MSELoss()  # MSE loss function
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
net, train_loss = train_network(model=net, n_epoch=200, batch_size=400,
                                train_data=bmap_np, train_labels=func_scores_np,
                                optimizer=optimizer, criterion=criterion,
                                get_train_loss=True)

# Make Predictions
y_pred = net(bmap_torch)
net_loss = criterion(y_pred.squeeze(), func_scores_torch)
print('MSE of PyTorch Network: {:.3f}'.format(net_loss.item()))
