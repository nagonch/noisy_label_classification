import torch
from data import FashionMNIST5, FashionMNIST6, CIFAR
from model import FCN, LeNet
from torch.utils.data import DataLoader
from torcheval.metrics.classification import (
    MulticlassRecall,
    MulticlassPrecision,
)
from torcheval.metrics import MulticlassF1Score
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def accuracy(result, answer, topk=1):
    # Source: https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    r"""
    result (batch_size, class_cnt)
    answer (batch_size)
    """
    # save the batch size before tensor mangling
    bz = answer.size(0)
    # ignore result values. its indices: (sz,cnt) -> (sz,topk)
    values, indices = result.topk(topk)
    # transpose the k best indice
    result = indices.t()  # (sz,topk) -> (topk, sz)

    # repeat same labels topk times to match result's shape
    answer = answer.view(1, -1)  # (sz) -> (1,sz)
    answer = answer.expand_as(result)  # (1,sz) -> (topk,sz)

    correct = result == answer  # (topk,sz) of bool vals
    correct = correct.flatten()  # (topk*sz) of bool vals
    correct = correct.float()  # (topk*sz) of 1s or 0s
    correct = correct.sum()  # counts 1s (correct guesses)
    correct = correct.mul_(1 / bz)  # convert into decimal

    return correct.item()


def eval(model_path, dataset_name):
    dataset_name_to_object = {
        "CIFAR": CIFAR,
        "FashionMNIST5": FashionMNIST5,
        "FashionMNIST6": FashionMNIST6,
    }
    dataset = dataset_name_to_object[dataset_name](train=False)
    if dataset_name == "CIFAR":
        model = LeNet(
            3,
        ).to(device)
    else:
        model = FCN(dataset[0][0].shape[0], 3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_data = DataLoader(dataset, batch_size=100, shuffle=False)
    y_preds = []
    p_preds = []
    y_gt = []
    for X, y in test_data:
        outputs = model(X)
        p_preds.append(outputs)
        y_preds.append(outputs.argmax(axis=1))
        y_gt.append(y)
    y_preds = torch.cat(y_preds)
    y_gt = torch.cat(y_gt)
    p_preds = torch.cat(p_preds)

    recall = MulticlassRecall(num_classes=3, average="macro")
    recall.update(y_preds, y_gt)
    print(f"recall: {recall.compute()}")

    precision = MulticlassPrecision(num_classes=3, average="macro")
    precision.update(y_preds, y_gt)
    print(f"precision: {precision.compute()}")

    f1 = MulticlassF1Score(num_classes=3, average="macro")
    f1.update(y_preds, y_gt)
    print(f"f1: {f1.compute()}")

    top1_acc_val = accuracy(p_preds, y_gt)
    print(f"top1 accuracy: {top1_acc_val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model-path", type=str, help="Path to the evaluated model"
    )
    parser.add_argument("-dataset-name", type=str, help="Name of the dataset")
    args = parser.parse_args()

    print(eval(args.model_path, args.dataset_name))
