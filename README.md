# torch-dms

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## What is this?

Pytorch - Deep Mutational Scanning (torch-dms) is a small python package made to train neural networks on 
amino-acid substitution data, predicting some chosen functional score(s).
We use the binary encoding of variants using
[BinaryMap Object](https://jbloomlab.github.io/dms_variants/dms_variants.binarymap.html)
as input to vanilla feed-forward networks.

## How do I install it?

To install the API and command-line scripts at the moment,
it suggested you clone the repository, create a conda
environment from `environment.yaml`, and run the tests to make
sure everything is working properly.

```
git clone git@github.com:matsengrp/torchdms.git
conda env create -f environment.yaml #follow prompts
conda activate dms
```

Install with `pip install -e .`

## CLI

`torch-dms` uses
[click](https://click.palletsprojects.com/en/7.x/) as a CLI manager. This means
that torch-dms has nested help pages for each command available.
```
 $ tdms -h
```

```
Usage: tdms [OPTIONS] COMMAND [ARGS]...

  A generalized method to train neural networks on deep mutational scanning
  data.

Options:
  -h, --help  Show this message and exit.

Commands:
  beta     This command will plot the beta coeff for each possible mutation
           at each site along the sequence as a heatmap

  contour  Evaluate the the latent space of a model with a two     dimensional
           latent space by predicting across grid of values

  create   Create a Model
  eval     Evaluate the performance of a model and dump     the a dictionary
           containing the results

  prep     Prepare a dataframe with aa subsitutions and targets in the
           format needed to present to a neural network.

  scatter  Evaluate and produce scatter plot of observed vs. predicted
           targets on the test set provided.

  train    Train a Model
```

## Example

Synopsis:

    tdms prep tstarr_dms_full.pkl NIH_PREP.pkl affinity_score expr_score
    tdms create NIH_PREP.pkl byov_10_10_10.model BuildYourOwnVanillaNet 10 10 10
    tdms train byov_10_10_10.model NIH_PREP.pkl --epochs 100
    tdms eval byov.model NIH_prep.pkl --scatter-plot-out NIH_Scatter_out.pdf

