import numpy as np
from torch.utils.data import Dataset as TorchDataset
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset(TorchDataset):
    def __init__(self, path, train=True):
        self.path = path
        self.T = None
        if train:
            self.Xs = torch.tensor(np.load(f"{self.path}/Xtr.npy"))
            self.Ys = torch.tensor(np.load(f"{self.path}/Str.npy"))
        else:
            self.Xs = torch.tensor(np.load(f"{self.path}/Xts.npy"))
            self.Ys = torch.tensor(np.load(f"{self.path}/Yts.npy"))
        self.Xs = self.Xs.float()
        self.Xs = self.Xs.to(device)
        self.Ys = self.Ys.to(device)

    def __len__(self):
        return len(self.Xs)


class FashionMNIST5(Dataset):
    def __init__(self, train=True):
        super().__init__("datasets/FashionMNIST0.5", train)
        self.T = torch.tensor(
            [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
        )

    def __getitem__(self, i):
        return self.Xs[i][None], self.Ys[i]


class FashionMNIST6(Dataset):
    def __init__(self, train=True):
        super().__init__("datasets/FashionMNIST0.6", train)
        self.path = "datasets/FashionMNIST0.6"
        self.T = torch.tensor(
            [[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]]
        )

    def __getitem__(self, i):
        return self.Xs[i][None], self.Ys[i]


class CIFAR(Dataset):
    def __init__(self, train=True):
        super().__init__("datasets/CIFAR", train)

    def __getitem__(self, i):
        return torch.permute(self.Xs[i], (-1, 0, 1)), self.Ys[i]


if __name__ == "__main__":
    pass
    # from matplotlib import pyplot as plt

    # one = FashionMNIST6()
    # print(one[0][0].shape)
    # for i in range(len(one)):
    #     plt.imshow(one[i][0], cmap="gray")
    #     plt.show()
    #     plt.close()
