import os

import torch
import torch.nn as nn
from torchvision import models

from models.neuroflux import NeurofluxModel

class ModelFactory:
    def __init__(self, model_name, num_classes, freeze_layers=True, hyperparams=None):
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_layers = freeze_layers
        self.hyperparams = hyperparams

    def get_model(self):
        """
        Supported models: 'resnet50', 'neuroflux'
        """
        if self.model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            if self.freeze_layers:
                # Freeze most layers but unfreeze the last few blocks for fine-tuning
                ct = 0
                for child in model.children():
                    ct += 1
                    if ct < 7:  # Freeze layers before layer4 in ResNet
                        for param in child.parameters():
                            param.requires_grad = False

            # Replace the final layer with proper weight initialization
            in_features = model.fc.in_features
            hidden_size = self.hyperparams.get("hidden_size", 512) if self.hyperparams else 512
            dropout_rate = self.hyperparams.get("dropout_rate", 0.6) if self.hyperparams else 0.6
            
            model.fc = nn.Sequential(
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_size, self.num_classes),
            )

        elif self.model_name == "neuroflux":
            model = NeurofluxModel(self.num_classes, self.hyperparams)

        else:
            raise ValueError(
                f"Unsupported model: {self.model_name}, please check the model_name in the config file"
            )

        return model


    def load_model(self, model_save_path):
        self.model.load_state_dict(torch.load(model_save_path))
        return self.model


    def save_model(model, model_save_path):
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)