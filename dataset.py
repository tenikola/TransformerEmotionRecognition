import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {
        'data': torch.tensor(self.data[idx], dtype=torch.float32),  # Assuming 'data' is a NumPy array
        'label': torch.tensor(self.labels[idx], dtype=torch.long)   # Assuming 'labels' is a list or array
    }
        return sample