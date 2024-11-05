import torch
from torch.utils.data import Dataset


class PermeabilityDataset(Dataset):
    """Custom dataset class for permeability data"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
