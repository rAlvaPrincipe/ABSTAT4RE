from torch.utils.data import Dataset

class REDataset(Dataset):
<<<<<<< HEAD
    def __init__(self, encodings, attn_masks, vec_labels, labels):
        self.encodings = encodings
        self.attn_masks = attn_masks
        self.vec_labels = vec_labels
        self.labels = labels
        
    def __len__(self):
        return self.vec_labels.shape[0]
        
    def __getitem__(self, idx):
        return self.encodings[idx], self.attn_masks[idx], self.vec_labels[idx], self.labels[idx]
=======
    def __init__(self, encodings, attn_masks, properties):
        self.encodings = encodings
        self.attn_masks = attn_masks
        self.properties = properties
        
    def __len__(self):
        return self.properties.shape[0]
        
    def __getitem__(self, idx):
        return self.encodings[idx], self.attn_masks[idx], self.properties[idx]
>>>>>>> 1e0355768f8c57bc49c47f74083f468431dcd2e9

