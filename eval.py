import torch
from data import FashionMNIST5, FashionMNIST6, CIFAR
from model import ResnetPretrained
from torch.utils.data import DataLoader
from torcheval.metrics.classification import (
    MulticlassRecall,
    MulticlassPrecision,
)
from torcheval.metrics import MulticlassF1Score
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval(model_path, dataset_name):
    dataset_name_to_object = {
        "CIFAR": CIFAR,
        "FashionMNIST5": FashionMNIST5,
        "FashionMNIST6": FashionMNIST6,
    }
    dataset = dataset_name_to_object[dataset_name](train=False)
    model = ResnetPretrained(dataset[0][0].shape[0], 3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_data = DataLoader(dataset, batch_size=100, shuffle=False)
    y_preds = []
    y_gt = []
    for X, y in test_data:
        outputs = model(X)
        y_preds.append(outputs.argmax(axis=1))
        y_gt.append(y)
    y_preds = torch.cat(y_preds)
    y_gt = torch.cat(y_gt)

    recall = MulticlassRecall(num_classes=3)
    recall.update(y_preds, y_gt)
    print(f"recall: {recall.compute()}")

    precision = MulticlassPrecision(num_classes=3)
    precision.update(y_preds, y_gt)
    print(f"precision: {precision.compute()}")

    f1 = MulticlassF1Score(num_classes=3)
    f1.update(y_preds, y_gt)
    print(f"f1: {f1.compute()}")

    top1_acc_val = (y_preds == y_gt).to(torch.float32).mean()
    print(f"top1 accuracy: {top1_acc_val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model-path", type=str, help="Path to the evaluated model"
    )
    parser.add_argument("-dataset-name", type=str, help="Name of the dataset")
    args = parser.parse_args()

    print(eval(args.model_path, args.dataset_name))
