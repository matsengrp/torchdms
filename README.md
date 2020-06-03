# torchdms

![build and test](https://github.com/matsengrp/torchdms/workflows/build%20and%20test/badge.svg)
[![Docker Repository on Quay](https://quay.io/repository/matsengrp/torchdms/status "Docker Repository on Quay")](https://quay.io/repository/matsengrp/torchdms)


## What is this?

PyTorch - Deep Mutational Scanning (`torchdms`) is a Python package made to train neural networks on amino-acid substitution data, predicting some chosen functional score(s).
We use the binary encoding of variants using [BinaryMap Object](https://jbloomlab.github.io/dms_variants/dms_variants.binarymap.html) as input to feed-forward networks.


## How do I install it?

    git clone git@github.com:matsengrp/torchdms.git
    cd torchdms
    pip install -r requirements.txt
    pip install .
    make test


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
