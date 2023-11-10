from torch import nn as nn
import torch.nn.functional as F
import torch


# Define FCN model (for FashionMNIST5 and FashionMNIST6 usage)
class FCN(nn.Module):
    def __init__(self, in_features, n_classes, hidden_size=128, return_logits=False):
        super().__init__()
        self.linear_in = nn.Linear(
            in_features,
            hidden_size,
        )
        self.hidden_1 = nn.Linear(
            hidden_size,
            hidden_size,
        )
        self.hidden_2 = nn.Linear(
            hidden_size,
            hidden_size,
        )
        self.out = nn.Linear(hidden_size, n_classes)
        self.return_logits = return_logits

    def forward(self, x):
        x = F.dropout(F.relu(self.linear_in(x)))
        x = F.dropout(F.relu(self.hidden_1(x)))
        x = F.dropout(F.relu(self.hidden_2(x)))
        x = self.out(x)

        if self.return_logits:
            return x
        return F.softmax(x, dim=1)


# Define LeNet model (for FashionMNIST5 and FashionMNIST6 usage)
class LeNet(nn.Module):
    def __init__(self, out_classes, return_logits=False):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_classes)
        self.return_logits = return_logits

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        if self.return_logits:
            return x
        return F.softmax(x, dim=1)


if __name__ == "__main__":
    pass
