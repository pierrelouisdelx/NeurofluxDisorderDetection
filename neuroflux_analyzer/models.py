import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import os

def get_transfer_learning_model(model_name, num_classes, freeze_layers=True):
    """
    Loads a pre-trained model and replaces its classifier.
    Supported models: 'resnet18', 'resnet50'
    """
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if freeze_layers:
            ct = 0
            for child in model.children():
                ct += 1
                if ct < 7:  # Freeze layers before layer4 in ResNet
                    for param in child.parameters():
                        param.requires_grad = False

            for param in model.fc.parameters():
                param.requires_grad = True

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 120),
            nn.Linear(120, num_classes)
        )

    elif model_name == 'resnet50':
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
            nn.Dropout(p=0.5), 
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

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

    def forward(self, x):
        return self.model(x)