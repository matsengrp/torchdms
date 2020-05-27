# torchdms

[![Docker Repository on Quay](https://quay.io/repository/matsengrp/torchdms/status "Docker Repository on Quay")](https://quay.io/repository/matsengrp/torchdms)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## What is this?

Pytorch - Deep Mutational Scanning (`torchdms`) is a small Python package made to train neural networks on amino-acid substitution data, predicting some chosen functional score(s).
We use the binary encoding of variants using [BinaryMap Object](https://jbloomlab.github.io/dms_variants/dms_variants.binarymap.html) as input to feed-forward networks.


## How do I install it?

To install the API and command-line scripts at the moment, it suggested you clone the repository, create a conda environment from `environment.yaml`, and run the tests to make sure everything is working properly.

    git clone git@github.com:matsengrp/torchdms.git
    conda env create -f environment.yaml
    conda activate dms
    pytest

Install with `pip install -e .`


## CLI

The command line interface is called `tdms`, and has nested subcommands.
Run

    $ tdms -h

to get started.


## Example

Synopsis:

    tdms prep tstarr_dms_full.pkl NIH_PREP.pkl affinity_score expr_score
    tdms create NIH_PREP.pkl my.model VanillaGGE(10,sigmoid,10,relu)
    tdms train my.model NIH_PREP.pkl --epochs 100
    tdms evaluate my.model NIH_prep.pkl --scatter-plot-out NIH_Scatter_out.pdf
