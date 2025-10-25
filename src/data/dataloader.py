import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class FlashFloodDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.FloatTensor(self.X[idx])
        y = torch.FloatTensor([self.y[idx]])
        return X, y.squeeze()


def create_dataloaders(train_X_path, train_y_path, val_X_path, val_y_path,
                       test_X_path, test_y_path, batch_size=128, subsample_ratio=None):
    train_dataset = FlashFloodDataset(train_X_path, train_y_path)
    val_dataset = FlashFloodDataset(val_X_path, val_y_path)
    test_dataset = FlashFloodDataset(test_X_path, test_y_path)

    if subsample_ratio is not None and subsample_ratio < 1.0:
        train_size = len(train_dataset)
        subsample_size = int(train_size * subsample_ratio)
        indices = np.random.choice(train_size, subsample_size, replace=False)
        train_dataset = Subset(train_dataset, indices)
        print(f"Subsampled training data: {subsample_size}/{train_size} samples ({subsample_ratio*100}%)")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
