import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class PermeabilityDataset(Dataset):
    """Custom dataset class for permeability data"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_balanced_loader(dataset, batch_size):
    """Create a DataLoader with balanced classes"""
    labels = dataset.y
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
