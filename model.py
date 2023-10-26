from torchvision.models import resnet50, ResNet50_Weights
from torch import nn as nn
import torch.nn.functional as F


# class ResnetPretrained(nn.Module):
#     def __init__(self, input_channels, out_classes):
#         super().__init__()
#         self.in_layer = nn.Conv2d(input_channels, 3, 3)
#         self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
#         self.out_layer = nn.Linear(1000, out_classes)

#     def forward(self, x):
#         x = F.relu(self.in_layer(x))
#         x = F.relu(self.resnet(x))
#         x = self.out_layer(x)

#         return x


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

        return x


if __name__ == "__main__":
    pass
    # model = ResnetPretrained(1, 3)
    # x = torch.rand(2, 1, 35, 35)
    # print(model(x).shape)
