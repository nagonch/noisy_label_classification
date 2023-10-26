import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ResnetPretrained
from tqdm import tqdm
from data import CIFAR, FashionMNIST5, FashionMNIST6
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model,
    train_dataloader,
    n_epochs,
    transition_matrix,
    forward_correction,
    lr=1e-4,
    save_model=True,
):
    if forward_correction and transition_matrix is not None:
        inv_transition = torch.linalg.inv(transition_matrix).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(n_epochs):
        for X, y in tqdm(train_dataloader, leave=False):
            optimizer.zero_grad()
            outputs = model(X)
            if forward_correction:
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                outputs = (inv_transition @ outputs.T).T
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item()}")
        if save_model:
            torch.save(model.state_dict(), "CIFAR_naive.pth")
    return model


def run(dataset_name, n_epochs, batch_size, lr=1e-4, save_model=False):
    dataset_name_to_object = {
        "CIFAR": CIFAR,
        "FashionMNIST5": FashionMNIST5,
        "FashionMNIST6": FashionMNIST6,
    }
    dataset = dataset_name_to_object[dataset_name]()
    training_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ResnetPretrained(dataset[0][0].shape[0], 3).to(device)
    train(
        model,
        training_data,
        n_epochs,
        dataset.T,
        False,
        lr=lr,
        save_model=save_model,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset-name", type=str, help="Name of the dataset")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--save-model", action="store_true", help="Save the model"
    )

    args = parser.parse_args()
    run(
        args.dataset_name,
        args.epochs,
        args.batch_size,
        lr=args.learning_rate,
        save_model=args.save_model,
    )
