import pickle
import torch
import torch.nn as nn


def from_pickle_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def to_pickle_file(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


class DataFactory:
    def __init__(self, bmap):
        self.binary_variants_array = bmap.binary_variants.toarray()
        self.bmap = bmap

    def feature_count(self):
        return self.binary_variants_array.shape[1]

    def nvariants(self):
        return self.bmap.nvariants

    def data_of_idxs(self, idxs):
        batch_X = torch.from_numpy(self.binary_variants_array[idxs]).float()
        batch_y = torch.from_numpy(self.bmap.func_scores[idxs]).float()
        batch_var = torch.from_numpy(self.bmap.func_scores_var[idxs]).float()
        return batch_X, batch_y, batch_var
