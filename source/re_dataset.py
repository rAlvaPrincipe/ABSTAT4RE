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

        self.attribute_tensor = self.compute_attribute_tensor(rel_labels)

    
    def compute_attribute_tensor(self, rel_labels):
        attribute_tensor = torch.zeros(len(rel_labels))
        for i, lab in enumerate(rel_labels):
            if lab == 'no_relation':
                attribute_tensor[i] = 1
        return attribute_tensor

    def __len__(self):
        return self.vec_labels.shape[0]


    def __getitem__(self, idx):
        return self.encodings[idx], self.attn_masks[idx], self.vec_labels[idx], self.rel_labels[idx], self.prop_labels[idx], self.integer_labels[idx], self.rel_labels[idx] == 'no_relation', self.attribute_tensor[idx]


    def get_integer_labels(self, prop_labels, mappings):
        # to garantee a index based on
        mappings = dict(sorted(mappings.items(), key=lambda item: item[1]))
        vals = list(mappings.values())

        integer_labels = list()
        for label in prop_labels:
            integer_labels.append(vals.index(label))

        return integer_labels