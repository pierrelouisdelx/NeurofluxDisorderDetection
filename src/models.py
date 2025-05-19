import torch.nn as nn
from torchvision import models
import torch
import os
from torch.nn import functional as F

def get_model(model_name, num_classes, freeze_layers=True):
    """
    Supported models: 'resnet50', 'efficientnet_b7', 'neuroflux_model'
    """
    if model_name == 'resnet50':
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
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(512, num_classes)
        )

    elif model_name == 'efficientnet_b7':
        model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)

        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False

        in_features = model.classifier[3].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == 'neuroflux':
        model = NeurofluxModel(num_classes)
                
    else:
        raise ValueError(f"Unsupported model: {model_name}, please check the model_name in the config file")

    return model

def load_model(model, model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    return model

def save_model(model, model_save_path):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

class NeurofluxModel(nn.Module):
    def __init__(self, num_classes):
        super(NeurofluxModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # -> [B, 16, 112, 112]
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # -> [B, 32, 56, 56]
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> [B, 64, 28, 28]
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # -> [B, 128, 14, 14]
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 56, 56]
        x = self.pool(F.relu(self.conv3(x)))  # [B, 64, 28, 28]
        x = self.pool(F.relu(self.conv4(x)))  # [B, 128, 14, 14]
        x = self.flatten(x)                                            # [B, ?]
        x = F.relu(self.fc1(x))             # [B, 128]
        x = self.fc2(x)                                               # [B, num_classes]
        return x
