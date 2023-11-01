import torch
from model import LeNet, FCN
from data import CIFAR, FashionMNIST5, FashionMNIST6
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from eval import topk_accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model,
    train_dataloader,
    n_epochs,
):
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-5,
    )

    model.train()

    for _ in tqdm(range(n_epochs)):
        for X, y in train_dataloader:
            optimizer.zero_grad()
            output_probabilities = model(X)
            loss = F.cross_entropy(output_probabilities, y)
            loss.backward()
            optimizer.step()
    return model


def val(
    models,
    val_dataloader,
):
    models_to_accs = []
    for model in models:
        model.eval()


if __name__ == "__main__":
    dataset_name = "FashionMNIST5"

    dataset_name_to_object = {
        "CIFAR": CIFAR,
        "FashionMNIST5": FashionMNIST5,
        "FashionMNIST6": FashionMNIST6,
    }
    train_dataset = dataset_name_to_object[dataset_name]()

    train_batch_size = 1280
    T_estimating_dataset_frac = 1

    T_estimating_size = int(len(train_dataset) * T_estimating_dataset_frac)

    if dataset_name == "CIFAR":
        model = LeNet(
            3,
        ).to(device)
    else:
        model = FCN(train_dataset[0][0].shape[0], 3).to(device)
    training_data = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    model = train(
        model,
        train_dataloader=training_data,
        n_epochs=10,
    )
    model.eval()
    t_estimating_dataset, _ = random_split(
        train_dataset,
        [T_estimating_size, len(train_dataset) - T_estimating_size],
    )
    t_estimating_dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=True
    )

    probs_0 = []
    probs_1 = []
    probs_2 = []

    for data, targets in t_estimating_dataloader:
        with torch.no_grad():
            P = F.softmax(model(data), dim=1)
        probs_0 += P[:, 0].tolist()
        probs_1 += P[:, 1].tolist()
        probs_2 += P[:, 2].tolist()

    probs_0 = sorted(probs_0)
    probs_1 = sorted(probs_1)
    probs_2 = sorted(probs_2)
    print(probs_0[:10], probs_0[-10:])
    print(probs_1[:10], probs_1[-10:])
    print(probs_2[:10], probs_2[-10:])
    # result_T = torch.zeros((3, 3))
    # X = next(iter(t_estimating_dataloader))[0]
    # with torch.no_grad():
    #     P = model(X)
    # for i in range(3):
    #     x_best = P[torch.argmax(P[:, i]), :]
    #     for j in range(3):
    #         result_T[i][j] = x_best[j]
    # print(result_T)
    # result_T[0][0] = 1 - result_T[1][0] - result_T[2][0]
    # result_T[1][1] = 1 - result_T[0][1] - result_T[2][1]
    # result_T[2][2] = 1 - result_T[0][2] - result_T[1][2]
    # result_T /= result_T.sum(dim=1, keepdim=True)
    # print(result_T)
