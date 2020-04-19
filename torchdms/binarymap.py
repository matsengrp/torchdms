import pickle
import torch
import torch.nn as nn


def from_pickle_file(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def to_pickle_file(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


class DataFactory:
    def __init__(self, bmap):
        self.binary_variants_array = bmap.binary_variants.toarray()
        self.bmap = bmap
        self.X = torch.from_numpy(self.binary_variants_array).float()
        self.Y = torch.from_numpy(self.bmap.func_scores).float()

    def feature_count(self):
        return self.binary_variants_array.shape[1]

    def nvariants(self):
        return self.bmap.nvariants

    def data_of_idxs(self, idxs):
        return self.X[idxs], self.Y[idxs]
