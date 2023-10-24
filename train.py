import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ResnetPretrained
from tqdm import tqdm
from data import CIFAR, FashionMNIST5, FashionMNIST6


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_backward_correction(
    model,
    train_dataloader,
    n_epochs,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        for X, y in tqdm(train_dataloader, leave=False):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/100], Loss: {loss.item()}")
    return model


if __name__ == "__main__":
    dataset = CIFAR()
    training_data = DataLoader(dataset, batch_size=100, shuffle=True)
    model = ResnetPretrained(3, 3).to(device)
    train_backward_correction(model, training_data, 100)
