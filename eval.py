import torch
from data import FashionMNIST5, FashionMNIST6, CIFAR
from model import FCN, LeNet
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def metrics(preds, y, eps=1e-9):
    """
    1. Estimate precision, recall, accuracy@1 and f1 score
        by each class for a batch predictions
    2. Return average value by class
    """
    max_class = torch.max(y)
    recall_list = []
    precision_list = []
    f1_list = []
    top1_acc_list = []
    for i in range(max_class + 1):
        y_i = (y == i).to(torch.long)
        pred_i = (preds == i).to(torch.long)
        TP = torch.sum((y_i == 1) & (pred_i == 1)).item()
        FP = torch.sum((y_i == 0) & (pred_i == 1)).item()
        TN = torch.sum((y_i == 0) & (pred_i == 0)).item()
        FN = torch.sum((y_i == 1) & (pred_i == 0)).item()
        recall = TP / (TP + FN + eps)
        precision = TP / (TP + FP + eps)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(2 * precision * recall / (precision + eps + recall))
        top1_acc_list.append((TP + TN) / y_i.shape[0])
    return (
        np.mean(recall_list),
        np.mean(precision_list),
        np.mean(f1_list),
        np.mean(top1_acc_list),
    )


def eval(models_folder, dataset_name):
    """
    Evaluates models in model_folder for given dataset_name.
    Produces recall, precision, f1 score and top1 accuracy
        averaged over all classes and batches.
    """
    dataset_name_to_object = {
        "CIFAR": CIFAR,
        "FashionMNIST5": FashionMNIST5,
        "FashionMNIST6": FashionMNIST6,
    }
    dataset = dataset_name_to_object[dataset_name](train=False)
    if dataset_name == "CIFAR":
        # LeNet for CIFAR for better performance
        model = LeNet(
            3,
        ).to(device)
    else:
        # Standard fully-connected model for FashionMNIST datasets
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

        recall, precision, f1, top1_acc = metrics(y_preds, y_gt)

        recalls.append(recall)
        precisions.append(precision)
        f1_vals.append(f1)
        accuracy_vals.append(top1_acc)

    # Estimate mean of each metrics across all batches
    mean_acc = round(np.mean(accuracy_vals), 4)
    mean_rec = round(np.mean(recalls), 4)
    mean_prec = round(np.mean(precisions), 4)
    mean_f1 = round(np.mean(f1_vals), 4)

    # Estimate std of each metrics across all batches
    std_acc = round(np.std(accuracy_vals), 4)
    std_rec = round(np.std(recalls), 4)
    std_prec = round(np.std(precisions), 4)
    std_f1 = round(np.std(f1_vals), 4)

    # Print the metrics
    print(f"Average accuracy over {i+1} models: {mean_acc} +- {std_acc}")
    print(f"Average recalls over {i+1} models: {mean_rec} +- {std_rec}")
    print(f"Average precisions over {i+1} models: {mean_prec} +- {std_prec}")
    print(f"Average f1 over {i+1} models: {mean_f1} +- {std_f1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-models-folder", type=str, help="Path a folder with trained models"
    )
    parser.add_argument("-dataset-name", type=str, help="Name of the dataset")
    args = parser.parse_args()

    eval(args.models_folder, args.dataset_name)
