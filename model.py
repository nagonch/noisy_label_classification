from torch import nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self, in_features, n_classes, hidden_size=128):
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

    def forward(self, x):
        x = F.dropout(F.relu(self.linear_in(x)))
        x = F.dropout(F.relu(self.hidden_1(x)))
        x = F.dropout(F.relu(self.hidden_2(x)))
        x = self.out(x)

        return F.softmax(x, dim=1)


if __name__ == "__main__":
    pass
