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
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def topk_accuracy(result, answer, topk=1):
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


def eval(models_folder, dataset_name):
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
    recalls = []
    precisions = []
    f1_vals = []
    accuracy_vals = []
    for i, model_path in enumerate(sorted(os.listdir(models_folder))):
        model.load_state_dict(torch.load(f"{models_folder}/{model_path}"))
        model.eval()
        test_data = DataLoader(dataset, batch_size=100, shuffle=True)
        y_preds = []
        p_preds = []
        y_gt = []
        for X, y in test_data:
            with torch.no_grad():
                outputs = model(X)
            p_preds.append(outputs)
            y_preds.append(outputs.argmax(axis=1))
            y_gt.append(y)
        y_preds = torch.cat(y_preds)
        y_gt = torch.cat(y_gt)
        p_preds = torch.cat(p_preds)

        recall = MulticlassRecall(num_classes=3, average="macro")
        recall.update(y_preds, y_gt)
        if i == 0:
            print(f"Best model by val loss recall: {recall.compute()}")

        precision = MulticlassPrecision(num_classes=3, average="macro")
        precision.update(y_preds, y_gt)
        if i == 0:
            print(f"Best model by val loss precision: {precision.compute()}")

        f1 = MulticlassF1Score(num_classes=3, average="macro")
        f1.update(y_preds, y_gt)
        if i == 0:
            print(f"Best model by val loss f1: {f1.compute()}")

        top1_acc_val = topk_accuracy(p_preds, y_gt)
        if i == 0:
            print(f"Best model by val loss top1 accuracy: {top1_acc_val}")

        recalls.append(recall.compute())
        precisions.append(precision.compute())
        f1_vals.append(f1.compute())
        accuracy_vals.append(top1_acc_val)

    print(f"Average accuracy over {i+1} models: {np.mean(accuracy_vals)}")
    print(f"Average recalls over {i+1} models: {np.mean(recalls)}")
    print(f"Average precisions over {i+1} models: {np.mean(precisions)}")
    print(f"Average f1 over {i+1} models: {np.mean(f1_vals)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-models-folder", type=str, help="Path to the evaluated model")
    parser.add_argument("-dataset-name", type=str, help="Name of the dataset")
    args = parser.parse_args()

    eval(args.models_folder, args.dataset_name)
