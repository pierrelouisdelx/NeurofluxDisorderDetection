import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models


def get_model(model_name, num_classes, freeze_layers=True, hyperparams=None):
    """
    Supported models: 'resnet50', 'neuroflux'
    """
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        if freeze_layers:
            # Freeze most layers but unfreeze the last few blocks for fine-tuning
            ct = 0
            for child in model.children():
                ct += 1
                if ct < 7:  # Freeze layers before layer4 in ResNet
                    for param in child.parameters():
                        param.requires_grad = False

        # Replace the final layer with proper weight initialization
        in_features = model.fc.in_features
        hidden_size = hyperparams.get("hidden_size", 512) if hyperparams else 512
        dropout_rate = hyperparams.get("dropout_rate", 0.6) if hyperparams else 0.6
        
        model.fc = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, num_classes),
        )

    elif model_name == "neuroflux":
        model = NeurofluxModel(num_classes, hyperparams)

    else:
        raise ValueError(
            f"Unsupported model: {model_name}, please check the model_name in the config file"
        )

    return model


def load_model(model, model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    return model


def save_model(model, model_save_path):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)


class NeurofluxModel(nn.Module):
    def __init__(self, num_classes, hyperparams=None):
        super(NeurofluxModel, self).__init__()
        
        conv_channels = hyperparams.get("conv_channels", 32) if hyperparams else 32
        dropout_rate = hyperparams.get("dropout_rate", 0.5) if hyperparams else 0.5

        self.conv1 = nn.Conv2d(3, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(conv_channels*2, conv_channels*4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(conv_channels*4, conv_channels*8, kernel_size=3, padding=1)
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
