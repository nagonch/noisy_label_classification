import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import FCN, LeNet
from tqdm import tqdm
from data import CIFAR, FashionMNIST5, FashionMNIST6
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_backward_correction(
    model,
    train_dataloader,
    n_epochs,
    inv_transition,
    lr=1e-2,
    eps=1e-9,
):
    optimizer = optim.Adagrad(model.parameters(), lr=lr, lr_decay=1e-6)

    model.train()

    for _ in tqdm(range(n_epochs)):
        for X, y in train_dataloader:
            optimizer.zero_grad()
            output_probabilities = model(X)
            loss_per_label = -torch.log(output_probabilities + eps)
            loss_per_label = (inv_transition @ loss_per_label.T).T
            loss = torch.gather(loss_per_label, -1, y.unsqueeze(-1)).reshape(-1).mean()
            loss.backward()
            optimizer.step()
    return model


def run_backward_correction(
    dataset_name,
    exp_name,
    n_epochs,
    batch_size,
    lr=1e-2,
    save_model=False,
    n_splits=10,
    eps=1e-6,
):
    dataset_name_to_object = {
        "CIFAR": CIFAR,
        "FashionMNIST5": FashionMNIST5,
        "FashionMNIST6": FashionMNIST6,
    }
    dataset = dataset_name_to_object[dataset_name]()
    inv_transition = torch.linalg.inv(dataset.T).to(device)
    if dataset.T is None:
        raise RuntimeError("No transition matrix for backward correction")

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
            model = LeNet(
                3,
            ).to(device)
        else:
            model = FCN(dataset[0][0].shape[0], 3).to(device)
        model.train()
        model = train_backward_correction(
            model,
            training_data,
            n_epochs,
            inv_transition,
            lr=lr,
        )
        losses = []
        model.eval()
        for X, y in val_data:
            with torch.no_grad():
                output_probabilities = model(X)
            loss_per_label = -torch.log(output_probabilities + eps)
            loss_per_label = (inv_transition @ loss_per_label.T).T
            loss = torch.gather(loss_per_label, -1, y.unsqueeze(-1)).reshape(-1).mean()
            losses.append(loss)
        models.append([model, torch.mean(torch.tensor(losses))])
    models = sorted(models, key=lambda x: x[-1])
    if save_model:
        if not os.path.exists("exp_name"):
            os.mkdir("exp_name")
        for i, (model, loss_val) in enumerate(models):
            print(f"top {i} validation loss value: {loss_val}")
            torch.save(
                model.state_dict(),
                f"{exp_name}/top_{str(i).zfill(2)}.pth",
            )
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset-name", type=str, help="Name of the dataset")
    parser.add_argument("-exp-name", type=str, help="Experiment name to save the model")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-2, help="Learning rate"
    )
    parser.add_argument("--save-model", action="store_true", help="Save the model")

    args = parser.parse_args()
    run_backward_correction(
        args.dataset_name,
        args.exp_name,
        args.epochs,
        args.batch_size,
        lr=args.learning_rate,
        save_model=args.save_model,
    )
