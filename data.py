import numpy as np


class Dataset:
    """
    A dataset base class
    get_train_data : return train data (X, Y)
    get_test_data : return train data (X, Y)
    get_transition_matrix : return transition matrix T
    """

    def __init__(self):
        self.path = None
        self.T = None

    def get_train_data(self):
        X = np.load(f"{self.path}/Xtr.npy")
        Y = np.load(f"{self.path}/Str.npy")

        return X, Y

    def get_test_data(self):
        X = np.load(f"{self.path}/Xts.npy")
        Y = np.load(f"{self.path}/Yts.npy")

        return X, Y

    def get_transition_matrix(self):
        return self.T


class FashionMNIST5(Dataset):
    def __init__(self):
        super().__init__()
        self.path = "datasets/FashionMNIST0.5"
        self.T = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])


class FashionMNIST6(Dataset):
    def __init__(self):
        super().__init__()
        self.path = "datasets/FashionMNIST0.6"
        self.T = np.array([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]])


class CIFAR(Dataset):
    def __init__(self):
        super().__init__()
        self.path = "datasets/CIFAR"


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
