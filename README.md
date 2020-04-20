# torch-dms

Install with `pip install -e .`

Synopsis:

    tdms prep _ignore/gb1.pkl _ignore/gb1-prepped.pkl
    tdms create SingleSigmoidNet _ignore/gb1-prepped.pkl _ignore/gb1-model.pt
    tdms train --epochs 200 _ignore/gb1-model.pt _ignore/gb1-prepped.pkl _ignore/gb1-out
    tdms eval _ignore/gb1-model.pt _ignore/gb1-prepped.pkl _ignore/gb1-out
