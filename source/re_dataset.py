from torch.utils.data import Dataset

class REDataset(Dataset):
    def __init__(self, encodings, attn_masks, properties):
        self.encodings = encodings
        self.attn_masks = attn_masks
        self.properties = properties
        
    def __len__(self):
        return self.properties.shape[0]
        
    def __getitem__(self, idx):
        return self.encodings[idx], self.attn_masks[idx], self.properties[idx]

