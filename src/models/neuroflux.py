import torch.nn as nn
import torch.nn.functional as F


class NeurofluxModel(nn.Module):
    def __init__(self, num_classes, hyperparams=None):
        super(NeurofluxModel, self).__init__()

        conv_channels = hyperparams.get("conv_channels", 32) if hyperparams else 32
        dropout_rate = hyperparams.get("dropout_rate", 0.5) if hyperparams else 0.5

        self.conv1 = nn.Conv2d(3, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            conv_channels, conv_channels * 2, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            conv_channels * 2, conv_channels * 4, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv2d(
            conv_channels * 4, conv_channels * 8, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 56, 56]
        x = self.pool(F.relu(self.conv3(x)))  # [B, 64, 28, 28]
        x = self.pool(F.relu(self.conv4(x)))  # [B, 128, 14, 14]
        x = self.flatten(x)  # [B, ?]
        x = F.relu(self.fc1(x))  # [B, 128]
        x = self.dropout(x)
        x = self.fc2(x)  # [B, num_classes]
        return x
