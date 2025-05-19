import torch.nn as nn
from torchvision import models
import torch
import os

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
        self.num_classes = num_classes

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # You need to know the flattened size after the conv layers
        # Let's assume input is (3, 128, 128) for example
        # You can calculate this dynamically or hardcode for now
        self.flattened_size = 256 * 16 * 16  # Example, adjust as needed

        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, self.num_classes)

    def forward(self, xb):
        xb = self.network(xb)
        xb = xb.view(xb.size(0), -1)
        xb = self.fc1(xb)
        xb = self.relu1(xb)
        xb = self.fc2(xb)
        xb = self.relu2(xb)
        xb = self.fc3(xb)
        return xb
