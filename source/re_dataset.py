from torch.utils.data import Dataset

class REDataset(Dataset):
    def __init__(self, encodings, attn_masks, vec_labels, rel_labels, prop_labels):
        self.encodings = encodings
        self.attn_masks = attn_masks
        self.vec_labels = vec_labels
        self.rel_labels = rel_labels
        self.prop_labels = prop_labels
        
    def __len__(self):
        return self.vec_labels.shape[0]
        
    def __getitem__(self, idx):
        return self.encodings[idx], self.attn_masks[idx], self.vec_labels[idx], self.rel_labels[idx], self.prop_labels[idx]

