"""
WindDataset — PyTorch Dataset for cGAN training
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class WindDataset(Dataset):
    """
    Dataset for wind field super-resolution training.

    Parameters
    ----------
    X : np.ndarray (N, 22, 21)
        Normalized coarse wind fields
    Y : np.ndarray (N, 201, 201)
        Normalized fine wind fields
    C : np.ndarray (N, 4)
        Condition vectors [Vmax, RMW, Pmin, lat]
    """

    def __init__(self, X, Y, C):
        # Replace any NaN with 0
        X = np.nan_to_num(X, nan=0.0)
        Y = np.nan_to_num(Y, nan=0.0)
        C = np.nan_to_num(C, nan=0.5)

        # Add channel dimension
        self.X = torch.FloatTensor(X).unsqueeze(1)  # (N, 1, 22, 21)
        self.Y = torch.FloatTensor(Y).unsqueeze(1)  # (N, 1, 201, 201)
        self.C = torch.FloatTensor(C)               # (N, 4)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.C[idx]
