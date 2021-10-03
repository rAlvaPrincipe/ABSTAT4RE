from torch.utils.data import Dataset
import torch
import pandas as pd

class REDataset(Dataset):
    def __init__(self, encodings, attn_masks, vec_labels, rel_labels, prop_labels, mappings ):
        self.encodings = encodings
        self.attn_masks = attn_masks
        self.vec_labels = vec_labels
        self.rel_labels = rel_labels
        self.prop_labels = prop_labels
        self.integer_labels = self.get_integer_labels(prop_labels, mappings)

   
    def __len__(self):
        return self.vec_labels.shape[0]


    def __getitem__(self, idx):
        return self.encodings[idx], self.attn_masks[idx], self.vec_labels[idx], self.rel_labels[idx], self.prop_labels[idx], self.integer_labels[idx]


    def get_integer_labels(self, prop_labels, mappings):
        # to garantee a index based on
        mappings = dict(sorted(mappings.items(), key=lambda item: item[1]))
        vals = list(mappings.values())

        integer_labels = list()
        for label in prop_labels:
            integer_labels.append(vals.index(label))

        return integer_labels