# torchdms

## THIS PACKAGE IS CURRENTLY UNDER ACTIVE DEVELOPMENT. 
**We expect a more stable version with documentation soon. Stay tuned!**

[![build and test](https://github.com/matsengrp/torchdms/workflows/build%20and%20test/badge.svg)](https://github.com/matsengrp/torchdms/actions?query=workflow%3A%22build+and+test%22)

👉 [Command-line and API documentation](https://matsengrp.github.io/torchdms/) 👈


## What is this?

PyTorch for Deep Mutational Scanning (`torchdms`) is a Python package made to train neural networks on amino-acid substitution data, predicting some chosen functional score(s).
We use the binary encoding of variants from the [`binarymap`](https://jbloomlab.github.io/binarymap/binarymap.binarymap.html) package as input to feed-forward networks.


<!--
## How do I install it?

NOTE: revise after publishing to pypi
```bash
pip install git+https://github.com/matsengrp/torchdms.git
```
-->

## Developer install (suggested)

```bash
git clone git@github.com:matsengrp/torchdms.git
cd torchdms
make install
make test
```
