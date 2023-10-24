import numpy as np
from torch.utils.data import Dataset
import torch


class FashionMNIST5(Dataset):
    def __init__(self, train=True):
        super().__init__()
        self.path = "datasets/FashionMNIST0.5"
        self.T = torch.tensor(
            [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
        )
        if train:
            self.Xs = torch.tensor(np.load("Xtr.npy"))
            self.Ys = torch.tensor(np.load("Str.npy"))
        else:
            self.Xs = torch.tensor(np.load("Xts.npy"))
            self.Ys = torch.tensor(np.load("Yts.npy"))

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, i):
        return self.Xs[i], self.Ys[i]


class FashionMNIST6(Dataset):
    def __init__(self, train=True):
        super().__init__()
        self.path = "datasets/FashionMNIST0.6"
        self.T = torch.tensor(
            [[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]]
        )
        if train:
            self.Xs = torch.tensor(np.load("Xtr.npy"))
            self.Ys = torch.tensor(np.load("Str.npy"))
        else:
            self.Xs = torch.tensor(np.load("Xts.npy"))
            self.Ys = torch.tensor(np.load("Yts.npy"))

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, i):
        return self.Xs[i], self.Ys[i]


class CIFAR(Dataset):
    def __init__(self, train=True):
        super().__init__()
        self.path = "datasets/CIFAR"
        self.T = None
        if train:
            self.Xs = torch.tensor(np.load("Xtr.npy"))
            self.Ys = torch.tensor(np.load("Str.npy"))
        else:
            self.Xs = torch.tensor(np.load("Xts.npy"))
            self.Ys = torch.tensor(np.load("Yts.npy"))

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, i):
        return self.Xs[i], self.Ys[i]


if __name__ == "__main__":
    one = FashionMNIST5()
    print(one.get_test_data())
    print(one.get_test_data())

    two = FashionMNIST6()
    print(two.get_test_data())
    print(two.get_test_data())

    three = CIFAR()
    print(three.get_test_data())
    print(three.get_test_data())
