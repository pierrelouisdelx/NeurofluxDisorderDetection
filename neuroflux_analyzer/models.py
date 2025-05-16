import torch.nn as nn
from torchvision import models

def get_model(model_name, num_classes, freeze_feature_extractor=True):
    """
    Loads a pre-trained model and replaces its classifier.
    Supported models: 'efficientnet_b0', 'resnet50', 'densenet121', 'vgg16'
    """
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        if freeze_feature_extractor:
            for param in model.features.parameters():
                param.requires_grad = False

        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        if freeze_feature_extractor:
            for param in model.parameters():
                param.requires_grad = False
            
            # Unfreeze the final layer
            for param in model.fc.parameters():
                param.requires_grad = True

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        if freeze_feature_extractor:
            for param in model.features.parameters():
                param.requires_grad = False

        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        if freeze_feature_extractor:
            for param in model.features.parameters():
                param.requires_grad = False

        in_features = model.classifier[6].in_features
        freeze_feature_extractor(model)
        model.classifier[6] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model
