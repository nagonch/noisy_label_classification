import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import FCN
from tqdm import tqdm
from data import CIFAR, FashionMNIST5, FashionMNIST6
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model,
    training_method,
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
            if training_method == "backward_correction":
                loss_per_label = (inv_transition @ loss_per_label.T).T
            loss = (
                torch.gather(loss_per_label, -1, y.unsqueeze(-1))
                .reshape(-1)
                .mean()
            )
            loss.backward()
            optimizer.step()
    return model


def run(
    dataset_name,
    exp_name,
    n_epochs,
    batch_size,
    training_method,
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
    if training_method == "backward_correction" and (dataset.T is not None):
        inv_transition = torch.linalg.inv(dataset.T).to(device)
    elif training_method == "backward_correction" and not (
        dataset.T is not None
    ):
        raise RuntimeError("No transition matrix for backward correction")

    dataset_size = len(dataset)
    train_ratio = 0.8
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    models = []

    for i in range(n_splits):
        print(f"Training model {i}:")
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        training_data = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        model = FCN(dataset[0][0].shape[0], 3).to(device)
        model.train()
        model = train(
            model,
            training_method,
            training_data,
            n_epochs,
            inv_transition,
            lr=lr,
        )
        losses = []
        model.eval()
        for X, y in val_data:
            output_probabilities = model(X)
            loss_per_label = -torch.log(output_probabilities + eps)
            if training_method == "backward_correction":
                loss_per_label = (inv_transition @ loss_per_label.T).T
            loss = (
                torch.gather(loss_per_label, -1, y.unsqueeze(-1))
                .reshape(-1)
                .mean()
            )
            losses.append(loss)
        models.append([model, torch.mean(torch.tensor(losses))])
    models = sorted(models, key=lambda x: x[-1])
    if save_model:
        if not os.path.exists("model_weghts"):
            os.mkdir("model_weghts")
        for i, (model, loss_val) in enumerate(models):
            print(f"top {i} validation loss value: {loss_val}")
            torch.save(
                model.state_dict(),
                f"model_weghts/{exp_name}_top_{str(i).zfill(2)}.pth",
            )
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset-name", type=str, help="Name of the dataset")
    parser.add_argument(
        "-exp-name", type=str, help="Experiment name to save the model"
    )
    parser.add_argument(
        "--epochs", type=int, default=40, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size"
    )
    parser.add_argument(
        "--training-method",
        choices=[
            "backward_correction",
        ],
        default="backward_correction",
        type=str,
        help="Name of the dataset",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-2, help="Learning rate"
    )
    parser.add_argument(
        "--save-model", action="store_true", help="Save the model"
    )

    args = parser.parse_args()
    run(
        args.dataset_name,
        args.exp_name,
        args.epochs,
        args.batch_size,
        args.training_method,
        lr=args.learning_rate,
        save_model=args.save_model,
    )
