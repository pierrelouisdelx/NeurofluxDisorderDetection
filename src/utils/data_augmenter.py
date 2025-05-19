import os
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


class DataAugmenter:
    def __init__(self, seed=42):
        """
        Initialize the DataAugmenter.

        Args:
            seed (int): Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        # Define augmentation transforms
        self.augmentation_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=(-15, 15), scale=(0.9, 1.1)),
            ]
        )

    def balance_dataset(
        self,
        image_paths,
        labels,
        output_dir="output/augmented_data",
        target_samples_per_class=None,
    ):
        """
        Balance the dataset by augmenting minority classes.

        Args:
            image_paths (list): List of paths to the original images
            labels (list): List of corresponding labels
            output_dir (str): Directory to store augmented images
            target_samples_per_class (int): Target number of samples per class. If None, uses max class count

        Returns:
            tuple: (augmented_image_paths, augmented_labels)
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Create class subdirectories
        unique_classes = set(labels)
        for class_name in unique_classes:
            (output_path / class_name).mkdir(exist_ok=True)

        # Count samples per class
        class_counts = pd.Series(labels).value_counts()
        if target_samples_per_class is None:
            target_samples_per_class = class_counts.max()

        augmented_paths = []
        augmented_labels = []

        # Process each class
        for class_name in unique_classes:
            # Get indices for current class
            class_indices = [i for i, label in enumerate(labels) if label == class_name]
            current_count = len(class_indices)

            if current_count < target_samples_per_class:
                # Calculate how many augmented samples we need
                samples_needed = target_samples_per_class - current_count

                for i in range(samples_needed):
                    # Randomly select an image from the current class
                    source_idx = random.choice(class_indices)
                    source_path = image_paths[source_idx]

                    # Load and augment image
                    image = Image.open(source_path)
                    augmented_image = self.augmentation_transforms(image)

                    # Generate unique filename
                    base_name = Path(source_path).stem
                    new_filename = f"{base_name}_aug_{i}.png"
                    new_path = output_path / class_name / new_filename

                    # Save augmented image
                    augmented_image.save(new_path)

                    # Add to augmented dataset
                    augmented_paths.append(str(new_path))
                    augmented_labels.append(class_name)

        return augmented_paths, augmented_labels

    def merge_and_split_data(
        self,
        original_paths,
        original_labels,
        augmented_paths,
        augmented_labels,
        split_ratio=(0.7, 0.15, 0.15),
    ):
        """
        Merge original and augmented data and split into train/val/test sets.

        Args:
            original_paths (list): Original image paths
            original_labels (list): Original labels
            augmented_paths (list): Augmented image paths
            augmented_labels (list): Augmented labels
            split_ratio (tuple): Train/val/test split ratios

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        # Combine original and augmented data
        all_paths = original_paths + augmented_paths
        all_labels = original_labels + augmented_labels

        # Create DataFrame
        df = pd.DataFrame({"image_path": all_paths, "label": all_labels})

        # Shuffle DataFrame
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        # Calculate split indices
        n_samples = len(df)
        train_size = int(n_samples * split_ratio[0])
        val_size = int(n_samples * split_ratio[1])

        # Split data
        train_df = df[:train_size]
        val_df = df[train_size : train_size + val_size]
        test_df = df[train_size + val_size :]

        return train_df, val_df, test_df

    def process_and_balance_dataset(
        self,
        image_paths,
        labels,
        output_dir="output/augmented_data",
        target_samples_per_class=None,
    ):
        """
        Process and balance the dataset, handling both new and existing augmented data.

        Args:
            image_paths (list): List of paths to the original images
            labels (list): List of corresponding labels
            output_dir (str): Directory to store augmented images
            target_samples_per_class (int): Target number of samples per class

        Returns:
            tuple: (train_image_paths, train_labels, val_image_paths, val_labels, test_image_paths, test_labels)
        """
        if os.path.exists(output_dir):
            print(
                f"Output directory {output_dir} already exists. Loading existing augmented data..."
            )
            # Load existing augmented data
            augmented_paths = []
            augmented_labels = []
            for class_dir in os.listdir(output_dir):
                class_path = os.path.join(output_dir, class_dir)
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.endswith((".png", ".jpg", ".jpeg")):
                            augmented_paths.append(os.path.join(class_path, img_file))
                            augmented_labels.append(class_dir)
        else:
            # Apply data augmentation to balance the dataset
            print("Applying data augmentation to balance the dataset...")
            augmented_paths, augmented_labels = self.balance_dataset(
                image_paths,
                labels,
                output_dir=output_dir,
                target_samples_per_class=target_samples_per_class,
            )

        # Merge original and augmented data, then split into train/val/test
        train_df, val_df, test_df = self.merge_and_split_data(
            image_paths,
            labels,
            augmented_paths,
            augmented_labels,
            split_ratio=(0.7, 0.15, 0.15),
        )

        # Extract paths and labels from DataFrames
        train_image_paths = train_df["image_path"].tolist()
        train_labels = train_df["label"].tolist()
        val_image_paths = val_df["image_path"].tolist()
        val_labels = val_df["label"].tolist()
        test_image_paths = test_df["image_path"].tolist()
        test_labels = test_df["label"].tolist()

        return (
            train_image_paths,
            train_labels,
            val_image_paths,
            val_labels,
            test_image_paths,
            test_labels,
        )
