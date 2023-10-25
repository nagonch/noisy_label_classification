import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ResnetPretrained
from tqdm import tqdm
from data import CIFAR, FashionMNIST5, FashionMNIST6


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model,
    train_dataloader,
    n_epochs,
    transition_matrix,
    forward_correction,
):
    inv_transition = torch.linalg.inv(transition_matrix).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    for epoch in range(n_epochs):
        for X, y in tqdm(train_dataloader, leave=False):
            optimizer.zero_grad()
            outputs = model(X)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            if forward_correction:
                probabilities = (inv_transition @ probabilities.T).T
            loss = criterion(probabilities, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item()}")
        torch.save(model.state_dict(), "fashionmnist6_naive.pth")
    return model


if __name__ == "__main__":
    dataset = FashionMNIST5()
    training_data = DataLoader(dataset, batch_size=100, shuffle=True)
    model = ResnetPretrained(1, 3).to(device)
    train(model, training_data, 100, dataset.T, False)
