import numpy as np
from torch.utils.data import Dataset as TorchDataset
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset(TorchDataset):
    """
    Base class for dataset loading routine
    """

    def __init__(self, path, train=True):
        self.path = path
        self.T = None
        dataset = np.load(path)
        if train:
            # Train labels
            self.Xs = torch.tensor(dataset["Xtr"])
            self.Ys = torch.tensor(dataset["Str"])
        else:
            # Test labels
            self.Xs = torch.tensor(dataset["Xts"])
            self.Ys = torch.tensor(dataset["Yts"])
        self.Xs = self.Xs.float()
        # Standard image preprocessing
        self.Xs = (self.Xs - self.Xs.min()) / (self.Xs.max() - self.Xs.min())
        self.Xs = self.Xs.to(device)
        self.Ys = self.Ys.to(device)

    def __len__(self):
        return len(self.Xs)


class FashionMNIST5(Dataset):
    def __init__(self, train=True):
        super().__init__("datasets/FashionMNIST0.5.npz", train)
        # T value provided for the dataset
        self.T = torch.tensor([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]).to(
            device
        )

    def __getitem__(self, i):
        return self.Xs[i].reshape(-1), self.Ys[i]


class FashionMNIST6(Dataset):
    def __init__(self, train=True):
        super().__init__("datasets/FashionMNIST0.6.npz", train)
        # T value provided for the dataset
        self.T = torch.tensor([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]]).to(
            device
        )

    def __getitem__(self, i):
        return self.Xs[i].reshape(-1), self.Ys[i]


class CIFAR(Dataset):
    def __init__(self, train=True):
        super().__init__("datasets/CIFAR.npz", train)
        # T value estimated using anchor points method
        self.T = torch.tensor(
            [
                [0.36434639, 0.32895993, 0.30669368],
                [0.32278317, 0.3544308, 0.32278375],
                [0.31430557, 0.32124954, 0.36444489],
            ]
        )

    def __getitem__(self, i):
        # Permute to return the channels in correct order
        return torch.permute(self.Xs[i], (2, 0, 1)), self.Ys[i]


if __name__ == "__main__":
    pass
