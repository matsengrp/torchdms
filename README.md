# torchdms

[![build and test](https://github.com/matsengrp/torchdms/workflows/build%20and%20test/badge.svg)](https://github.com/matsengrp/torchdms/actions?query=workflow%3A%22build+and+test%22)


## What is this?

PyTorch - Deep Mutational Scanning (`torchdms`) is a Python package made to train neural networks on amino-acid substitution data, predicting some chosen functional score(s).
We use the binary encoding of variants using [BinaryMap Object](https://jbloomlab.github.io/dms_variants/dms_variants.binarymap.html) as input to feed-forward networks.


## How do I install it?

<!-- NOTE: revise after publishing to pypi -->
```bash
pip install git+https://github.com/matsengrp/torchdms.git
```

## Developer install

```bash
git clone git@github.com:matsengrp/torchdms.git
cd torchdms
pip install -r requirements.txt
pip install -e .
make test
```

## CLI

The command line interface is called `tdms`, and has nested subcommands.
Run
```bash
$ tdms -h
```

to get started.


## Example

Synopsis:

```bash
tdms prep tstarr_dms_full.pkl NIH_PREP.pkl affinity_score expr_score
tdms create NIH_PREP.pkl my.model VanillaGGE(10,sigmoid,10,relu)
tdms train my.model NIH_PREP.pkl --epochs 100
tdms evaluate my.model NIH_prep.pkl --scatter-plot-out NIH_Scatter_out.pdf
```
