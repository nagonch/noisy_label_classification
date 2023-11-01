import torch
from model import LeNet, FCN
from data import CIFAR, FashionMNIST5, FashionMNIST6
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from eval import topk_accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def train(model, train_dataloader, val_dataloader, n_epochs, lr=0.01):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for epoch in tqdm(range(n_epochs)):
        model.train()
        preds = []
        ys = []
        for X, y in train_dataloader:
            ys.append(y)
            optimizer.zero_grad()
            output_logits = model(X)
            preds.append(F.softmax(output_logits, dim=1))
            loss = F.cross_entropy(output_logits, y)
            loss.backward()
            optimizer.step()
        train_accuracy = topk_accuracy(torch.cat(preds), torch.cat(ys))
        print(f"Train accuracy epoch {epoch}: {train_accuracy}, max prob: {torch.cat(preds).max()}")
        with torch.no_grad():
            model.eval()
            val_preds = []
            val_ys = []
            for X, y in val_dataloader:
                val_ys.append(y)
                output_logits = model(X)
                val_preds.append(F.softmax(output_logits, dim=1))
            val_accuracy = topk_accuracy(torch.cat(val_preds), torch.cat(val_ys)) 
            print(f"Val accuracy epoch {epoch}: {val_accuracy}, max prob: {torch.cat(val_preds).max()}")
        print(loss)
        scheduler.step()
        
    return model


if __name__ == "__main__":
    # GLOBALS -------------------
    dataset_name = "CIFAR"
    train_frac = 0.75
    val_frac = 0.25
    train_batch_size = 128
    k_splits = 1
    n_epochs = 200
    lr = 1e-3
    # ---------------------------

    dataset_name_to_object = {
        "CIFAR": CIFAR,
        "FashionMNIST5": FashionMNIST5,
        "FashionMNIST6": FashionMNIST6,
    }
    dataset = dataset_name_to_object[dataset_name]()
    train_size = int(len(dataset) * train_frac)
    val_size = int(len(dataset) * val_frac)

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
            train_data, batch_size=train_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_data, batch_size=val_size, shuffle=True
        )
        model = train(
            model,
            train_dataloader=training_dataloader,
            val_dataloader=val_dataloader,
            n_epochs=n_epochs,
            lr=lr,
        )
        models.append(model)

    model = val(
        models,
        val_dataloader,
    )

    estimation_dataloader = DataLoader(
        train_data, batch_size=train_size, shuffle=True
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
