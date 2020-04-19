import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def from_pickle_file(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def to_pickle_file(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


class BinarymapDataset(Dataset):
    """Binarymap dataset."""

    def __init__(self, bmap):
        self.binary_variants_array = bmap.binary_variants.toarray()
        self.bmap = bmap
        self.variants = torch.from_numpy(self.binary_variants_array).float()
        self.func_scores = torch.from_numpy(self.bmap.func_scores).float()

    def __len__(self):
        return self.bmap.nvariants

    def __getitem__(self, idxs):
        return {"variants": self.variants[idxs], "func_scores": self.func_scores[idxs]}

    def feature_count(self):
        return self.binary_variants_array.shape[1]
