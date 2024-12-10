import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class PermeabilityDataset(Dataset):
    """Custom dataset class for permeability data"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_balanced_loader(dataset, batch_size):
    """Create a DataLoader with balanced class weights"""
    # Convert tensor to numpy for counting if necessary
    if torch.is_tensor(dataset.y):
        labels = dataset.y.numpy()
    else:
        labels = dataset.y

    labels = labels.astype(int)  # Convert to integers
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]

    # Convert weights to tensor
    weights = torch.DoubleTensor(weights)

    sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
