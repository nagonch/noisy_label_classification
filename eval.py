import torch
from data import FashionMNIST5, FashionMNIST6, CIFAR
from model import ResnetPretrained
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = ResnetPretrained(3, 3).to(device)
    model.load_state_dict(torch.load("CIFAR_naive.pth"))
    data = CIFAR(train=False)
    test_data = DataLoader(data, batch_size=100, shuffle=True)
    y_preds = []
    y_gt = []
    for X, y in test_data:
        y_preds.append(model(X).argmax(axis=1))
        y_gt.append(y)
    y_preds = torch.cat(y_preds)
    y_gt = torch.cat(y_gt)
    print((y_preds == y_gt).to(torch.float32).mean())
    # print(multiclass_f1_score(y_preds, y_gt, num_classes=3))
