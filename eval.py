import torch
from data import FashionMNIST5
from model import ResnetPretrained
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = ResnetPretrained(1, 3).to(device)
    model.load_state_dict(torch.load("fashionmnist5_naive.pth"))
    data = FashionMNIST5(train=False)
    test_data = DataLoader(data, batch_size=100, shuffle=True)
    y_preds = []
    y_gt = []
    for X, y in test_data:
        y_preds.append(model(X).argmax(axis=1))
        y_gt.append(y)
    print(y_preds)
