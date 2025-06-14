import argparse
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from dataset import NeurofluxDataset
from models.model_factory import ModelFactory
from training import train_model, evaluate_model
from utils.config_loader import ConfigLoader
from utils.data_augmenter import DataAugmenter
from utils.file_utils import get_images_and_labels
from utils.preprocessing import MRIPreprocessor
from utils.transforms import get_train_transforms, get_val_test_transforms

OUTPUT_DIR = "output"


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
    parser.add_argument(
        "mode",
        choices=["preprocess", "train", "evaluate", "predict"],
        help="Mode to run: 'preprocess', 'train', 'evaluate', or 'predict'",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="configs/dataset_config.json",
        help="Path to the dataset configuration JSON file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/resnet50_config.json",
        help="Path to the model configuration JSON file",
    )
    parser.add_argument(
        "--image_path", type=str, help="Path to the MRI scan image for prediction"
    )

    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup TensorBoard
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(OUTPUT_DIR, "runs", current_time)
    writer = SummaryWriter(log_dir=log_dir)

    dataset_cfg = ConfigLoader(args.dataset_config).config
    model_cfg = ConfigLoader(args.model_config).config

    print(model_cfg)

    set_seed(dataset_cfg.get("random_seed", 42))  # Set seed for reproducibility

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    image_paths, labels = get_images_and_labels(dataset_cfg.get("data_dir"))
    (
        train_image_paths,
        train_labels,
        val_image_paths,
        val_labels,
        test_image_paths,
        test_labels,
    ) = DataAugmenter().process_and_balance_dataset(image_paths, labels)

    print(f"Train set: {len(train_image_paths)} images")
    print(f"Validation set: {len(val_image_paths)} images")
    print(f"Test set: {len(test_image_paths)} images")

    # Print amount of each class in the training set
    train_labels_series = pd.Series(train_labels)
    print(f"Training set class distribution: {train_labels_series.value_counts()}")

    if args.mode == "preprocess":
        preprocessor = MRIPreprocessor()
        preprocessor.analyze_dataset(train_image_paths)
        preprocessor.detect_outliers(train_image_paths, train_labels)
        preprocessor.visualize_dataset_tsne(train_image_paths, train_labels)
        return

    # Load data
    train_dataset = NeurofluxDataset(
        train_image_paths,
        train_labels,
        transform=get_train_transforms(dataset_cfg.get("image_size")),
        class_names=dataset_cfg.get("class_names"),
    )
    val_dataset = NeurofluxDataset(
        val_image_paths,
        val_labels,
        transform=get_val_test_transforms(dataset_cfg.get("image_size")),
        class_names=dataset_cfg.get("class_names"),
    )
    test_dataset = NeurofluxDataset(
        test_image_paths,
        test_labels,
        transform=get_val_test_transforms(dataset_cfg.get("image_size")),
        class_names=dataset_cfg.get("class_names"),
    )

    # Calculate class weights for balanced sampling
    class_counts = pd.Series(train_labels).value_counts()
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=model_cfg.get("batch_size"), 
        sampler=sampler
    )
    val_loader = DataLoader(
        val_dataset, batch_size=model_cfg.get("batch_size"), shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=model_cfg.get("batch_size"), shuffle=False
    )

    # Load model
    model = ModelFactory(
        model_cfg.get("model_name"), len(dataset_cfg.get("class_names")), model_cfg.get("hyperparams")
    ).get_model()
    model.to(device)

    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=model_cfg.get("learning_rate"),
        weight_decay=model_cfg.get("weight_decay"),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=model_cfg.get("lr_factor"), patience=model_cfg.get("lr_patience")
    )

    if args.mode == "train":
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=model_cfg.get("num_epochs"),
            device=device,
            model_save_path=os.path.join(OUTPUT_DIR, model_cfg.get("model_save_path")),
            class_names=dataset_cfg.get("class_names"),
            writer=writer,
        )

        writer.close()

        # Evaluate on test set
        evaluate_model(model, test_loader, device, dataset_cfg.get("class_names"))

    elif args.mode == "evaluate":
        model = ModelFactory(
            model_cfg.get("model_name"), len(dataset_cfg.get("class_names"))
        ).load_model(os.path.join(OUTPUT_DIR, model_cfg.get("model_save_path")), model)
        evaluate_model(model, test_loader, device, dataset_cfg.get("class_names"))

    elif args.mode == "predict":
        if args.image_path is None:
            raise ValueError("Image path is required for prediction")

        model = ModelFactory(
            model_cfg.get("model_name"), len(dataset_cfg.get("class_names"))
        ).load_model(os.path.join(OUTPUT_DIR, model_cfg.get("model_save_path")), model)
        model.eval()
        with torch.no_grad():
            image = Image.open(args.image_path).convert("RGB")
            image = get_val_test_transforms(dataset_cfg.get("image_size"))(image)
            image = image.unsqueeze(0).to(device)

            outputs = model(image)
            _, predicted = outputs.max(1)

            print(
                f"Predicted class: {dataset_cfg.get('class_names')[predicted.item()]}"
            )

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
