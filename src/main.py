import argparse
import os
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from utils.config_loader import load_config
from utils.file_utils import get_images_and_labels
from models import get_transfer_learning_model, load_model, save_model, NeurofluxModel
from datasets import NeurofluxDataset
from utils.transforms import get_train_transforms, get_val_test_transforms
from training import train_model, evaluate_model
from utils.preprocessing import MRIPreprocessor
from utils.data_augmentation import process_and_balance_dataset

OUTPUT_DIR = 'output'

def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Neuroflux Disorder Phase Classifier")
    parser.add_argument('mode', choices=['preprocess', 'train', 'evaluate', 'predict', 'train_custom_model'], help="Mode to run: 'preprocess', 'train', 'evaluate', or 'predict'")
    parser.add_argument('--dataset_config', type=str, default='configs/dataset_config.json', help="Path to the dataset configuration JSON file")
    parser.add_argument('--model_config', type=str, default='configs/resnet50_config.json', help="Path to the model configuration JSON file")
    parser.add_argument('--image_path', type=str, help="Path to the MRI scan image for prediction")

    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset_cfg = load_config(args.dataset_config)
    model_cfg = load_config(args.model_config)

    set_seed(dataset_cfg.get('random_seed', 42)) # Set seed for reproducibility

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    image_paths, labels = get_images_and_labels(dataset_cfg.get('data_dir'))
    train_image_paths, train_labels, val_image_paths, val_labels, test_image_paths, test_labels = process_and_balance_dataset(image_paths, labels)

    print(f"Train set: {len(train_image_paths)} images")
    print(f"Validation set: {len(val_image_paths)} images")
    print(f"Test set: {len(test_image_paths)} images")

    # Print amount of each class in the training set
    train_labels_series = pd.Series(train_labels)
    print(f"Training set class distribution: {train_labels_series.value_counts()}")

    if args.mode == 'preprocess':
        preprocessor = MRIPreprocessor()

        # Analyze dataset
        preprocessor.analyze_dataset(train_image_paths)

        # Detect outliers
        preprocessor.detect_outliers(train_image_paths, train_labels)

        # Visualize dataset
        preprocessor.visualize_dataset_tsne(train_image_paths, train_labels)
        return

    # Calculate class weights for balanced sampling
    class_counts = train_labels_series.value_counts()
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True
    )

    # Load data
    train_dataset = NeurofluxDataset(train_image_paths, train_labels, transform=get_train_transforms(dataset_cfg.get('image_size')), class_names=dataset_cfg.get('class_names'))
    val_dataset = NeurofluxDataset(val_image_paths, val_labels, transform=get_val_test_transforms(dataset_cfg.get('image_size')), class_names=dataset_cfg.get('class_names'))
    test_dataset = NeurofluxDataset(test_image_paths, test_labels, transform=get_val_test_transforms(dataset_cfg.get('image_size')), class_names=dataset_cfg.get('class_names'))

    train_loader = DataLoader(train_dataset, batch_size=model_cfg.get('batch_size'), sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=model_cfg.get('batch_size'), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=model_cfg.get('batch_size'), shuffle=False)

    # Load model
    if args.mode == 'train_custom_model':
        model = NeurofluxModel(len(dataset_cfg.get('class_names')))
    else:
        model = get_transfer_learning_model(model_cfg.get('model_name'), len(dataset_cfg.get('class_names')))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    if args.mode == 'train' or args.mode == 'train_custom_model':
        # Train the model
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=30,
            device=device,
            model_save_path='output/best_model.pth',
            class_names=dataset_cfg.get('class_names')
        )

        # Evaluate on test set
        evaluate_model(model, test_loader, device, dataset_cfg.get('class_names'))

        # Save the model
        save_model(model, os.path.join(OUTPUT_DIR, model_cfg.get('model_save_path')))

    elif args.mode == 'evaluate':
        model = load_model(model, os.path.join(OUTPUT_DIR, model_cfg.get('model_save_path')))
        evaluate_model(model, test_loader, device, dataset_cfg.get('class_names'))

    elif args.mode == 'predict':
        if args.image_path is None:
            raise ValueError("Image path is required for prediction")

        model = load_model(model, os.path.join(OUTPUT_DIR, model_cfg.get('model_save_path')))
        model.eval()
        with torch.no_grad():
            image = Image.open(args.image_path).convert('RGB')
            image = get_val_test_transforms(dataset_cfg.get('image_size'))(image)
            image = image.unsqueeze(0).to(device)

            outputs = model(image)
            _, predicted = outputs.max(1)

            print(f"Predicted class: {dataset_cfg.get('class_names')[predicted.item()]}")

    else:
        raise ValueError(f"Invalid mode: {args.mode}")

if __name__ == '__main__':
    main()