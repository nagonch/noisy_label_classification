import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from itertools import chain
from model import FCN, LeNet
from data import CIFAR, FashionMNIST5, FashionMNIST6
import os
import argparse
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_co_teaching(
    model_f,
    model_g,
    train_dataloader,
    n_epochs,
    transition_matrix,
    lr=1e-2,
):
    """
    Implements the co-teaching training method (Bo Han et al., 2018)
    model_f: the first model
    model_g: the second model
    train_loader: training dataloader
    n_epochs: number of epochs
    transition_matrix: the noise matrix T
    lr: learning rate
    """
    epsilon = 1 - torch.mean(
        torch.diagonal(transition_matrix)
    )  # Calculating noise rate from the transition matrix
    R = 1  # initially train on 100% of samples
    epoch_k = 11  # epoch at which R stops reducing
    optimizer = optim.Adam(chain(model_f.parameters(), model_g.parameters()), lr=lr)

    model_f.train()
    model_g.train()

    for epoch in tqdm(range(n_epochs)):
        for X, y in train_dataloader:
            f_pred = model_f(X)
            g_pred = model_g(X)

            losses_f = F.cross_entropy(f_pred, y, reduction="none")
            losses_g = F.cross_entropy(g_pred, y, reduction="none")

            # get top R indices with smallest loss for each model
            _, f_indices = torch.topk(
                losses_f, k=int(int(losses_f.size(0)) * R), largest=False
            )
            _, g_indices = torch.topk(
                losses_g, k=int(int(losses_g.size(0)) * R), largest=False
            )

            # backpropogate only most certain sample losses through the peer model
            # (the co-teaching step)
            model_f_loss_filter = torch.zeros((losses_f.size(0))).cuda()
            model_f_loss_filter[g_indices] = 1.0
            losses_f = (model_f_loss_filter * losses_f).mean()

            model_g_loss_filter = torch.zeros((losses_g.size(0))).cuda()
            model_g_loss_filter[f_indices] = 1.0
            losses_g = (model_g_loss_filter * losses_g).mean()

            optimizer.zero_grad()
            losses_f.backward()
            optimizer.step()

            optimizer.zero_grad()
            losses_g.backward()
            optimizer.step()
        R = 1 - epsilon * min(
            epoch / epoch_k, 1
        )  # reducing R until threshold epoch epoch_k

    return model_f


def run_co_teaching(
    dataset_name,
    exp_name,
    n_epochs,
    batch_size,
    lr=1e-2,
    save_model=False,
    n_splits=10,
):
    dataset_name_to_object = {
        "CIFAR": CIFAR,
        "FashionMNIST5": FashionMNIST5,
        "FashionMNIST6": FashionMNIST6,
    }
    dataset = dataset_name_to_object[dataset_name]()

    dataset_size = len(dataset)
    train_ratio = 0.8
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    models = []

    for i in range(n_splits):
        print(f"Training model {i}:")
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        training_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        if dataset_name == "CIFAR":
            # LeNet for CIFAR for better performance
            model_f = LeNet(
                3,
            ).to(device)
            model_g = LeNet(
                3,
            ).to(device)
        else:
            # Standard fully-connected model for FashionMNIST datasets
            model_f = FCN(dataset[0][0].shape[0], 3).to(device)
            model_g = FCN(dataset[0][0].shape[0], 3).to(device)

        model_f.train()
        model_g.train()

        model = train_co_teaching(
            model_f,
            model_g,
            training_data,
            n_epochs,
            dataset.T,
            lr=lr,
        )
        losses = []
        model.eval()
        for X, y in val_data:
            with torch.no_grad():
                output_probabilities = model(X)
            loss = F.cross_entropy(output_probabilities, y)
            losses.append(loss)
        models.append([model, torch.mean(torch.tensor(losses))])
    models = sorted(models, key=lambda x: x[-1])
    if save_model:
        if not os.path.exists(exp_name):
            os.mkdir(exp_name)
        for i, (model, loss_val) in enumerate(models):
            torch.save(
                model.state_dict(),
                f"{exp_name}/top_{str(i).zfill(2)}.pth",
            )
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset-name", type=str, help="Name of the dataset")
    parser.add_argument("-exp-name", type=str, help="Experiment name to save the model")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=5e-4, help="Learning rate"
    )
    parser.add_argument("--save-model", action="store_true", help="Save the model")

    args = parser.parse_args()
    run_co_teaching(
        args.dataset_name,
        args.exp_name,
        args.epochs,
        args.batch_size,
        lr=args.learning_rate,
        save_model=args.save_model,
    )
