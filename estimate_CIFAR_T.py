import torch
from model import LeNet
from data import CIFAR
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model,
    train_dataloader,
    n_epochs,
    lr=1e-2,
):
    optimizer = optim.Adagrad(model.parameters(), lr=lr, lr_decay=1e-6)

    model.train()

    for _ in tqdm(range(n_epochs)):
        for X, y in train_dataloader:
            optimizer.zero_grad()
            output_probabilities = model(X)
            loss = F.cross_entropy(output_probabilities, y)
            loss.backward()
            optimizer.step()
    return model


if __name__ == "__main__":
    train_batch_size = 128
    T_estimating_dataset_frac = 0.1

    train_dataset = CIFAR()
    T_estimating_size = int(len(train_dataset) * T_estimating_dataset_frac)

    model = LeNet(
        3,
    ).to(device)
    training_data = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    model = train(
        model,
        train_dataloader=training_data,
        n_epochs=40,
        lr=1e-3,
    )

    t_estimating_dataset, _ = random_split(
        train_dataset,
        [T_estimating_size, len(train_dataset) - T_estimating_size],
    )
    t_estimating_dataloader = DataLoader(
        train_dataset, batch_size=T_estimating_size, shuffle=True
    )
    result_T = torch.zeros((3, 3))
    X = next(iter(t_estimating_dataloader))[0]
    P = model(X)
    for i in range(3):
        x_best = P[torch.argmax(P[:, i]), :]
        for j in range(3):
            result_T[i][j] = x_best[j]
    result_T /= result_T.sum(dim=1, keepdim=True)
    print(result_T)
