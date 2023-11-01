import torch
from model import LeNet, FCN
from data import CIFAR, FashionMNIST5, FashionMNIST6
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from eval import topk_accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_dataloader, n_epochs, lr=0.01):
    optimizer = optim.Adagrad(model.parameters(), lr=lr, lr_decay=1e-6)

    model.train()

    for _ in tqdm(range(n_epochs)):
        for X, y in train_dataloader:
            optimizer.zero_grad()
            output_logits = model(X)
            loss = F.cross_entropy(output_logits, y)
            loss.backward()
            optimizer.step()
    return model


def val(
    models,
    val_dataloader,
):
    models_to_accs = []
    for i, model in enumerate(models):
        model.eval()
        accs = []
        for X, y in val_dataloader:
            p_pred = model(X)
            accs.append(topk_accuracy(p_pred, y))
        acc = torch.mean(torch.tensor(accs))
        print(f"model {i} accuracy: {acc}")
        models_to_accs.append([model, acc])

    models_to_accs = sorted(models_to_accs, key=lambda x: -x[-1])
    print(f"best model accuarcy: {models_to_accs[0][1]}")
    return models_to_accs[0][0]


if __name__ == "__main__":
    # GLOBALS -------------------
    dataset_name = "FashionMNIST5"
    train_frac = 0.75
    val_frac = 0.25
    train_batch_size = 128
    k_splits = 3
    n_epochs = 100
    lr = 1e-2
    # ---------------------------

    dataset_name_to_object = {
        "CIFAR": CIFAR,
        "FashionMNIST5": FashionMNIST5,
        "FashionMNIST6": FashionMNIST6,
    }
    dataset = dataset_name_to_object[dataset_name]()
    train_size = int(len(dataset) * train_frac)
    val_size = int(len(dataset) * val_frac)

    estimation_dataloader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=True
    )

    models = []

    for i in range(k_splits):
        if dataset_name == "CIFAR":
            model = LeNet(
                3,
                return_logits=True,
            ).to(device)
        else:
            model = FCN(
                dataset[0][0].shape[0],
                3,
                return_logits=True,
            ).to(device)

        train_data, val_data = random_split(
            dataset,
            [train_size, val_size],
        )

        training_dataloader = DataLoader(
            train_data, batch_size=train_batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_data, batch_size=train_batch_size, shuffle=True
        )
        model = train(
            model,
            train_dataloader=training_dataloader,
            n_epochs=n_epochs,
            lr=lr,
        )
        models.append(model)

    model = val(
        models,
        val_dataloader,
    )

    model.eval()
    result_T = torch.zeros((3, 3))
    X = next(iter(estimation_dataloader))[0]
    with torch.no_grad():
        P = F.softmax(model(X), dim=1)
    for i in range(3):
        x_best = P[torch.argmax(P[:, i]), :]
        for j in range(3):
            result_T[i][j] = x_best[j]
    result_T /= result_T.sum(dim=1, keepdim=True)
    print(result_T)
