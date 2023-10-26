import torch
from model import FCN
from data import CIFAR
from train import train
from torch.utils.data import DataLoader, random_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    train_batch_size = 128
    T_estimating_dataset_frac = 0.1

    train_dataset = CIFAR()
    T_estimating_size = int(len(train_dataset) * T_estimating_dataset_frac)

    model = FCN(train_dataset[0][0].shape[0], 3).to(device)
    training_data = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    model = train(
        model,
        training_method=None,
        train_dataloader=training_data,
        n_epochs=40,
        inv_transition=None,
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
    print(result_T)
