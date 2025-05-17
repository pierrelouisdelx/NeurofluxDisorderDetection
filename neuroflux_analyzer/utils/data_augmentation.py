import os
from pathlib import Path
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import random

def balance_dataset_with_augmentation(
    image_paths,
    labels,
    output_dir='new_aug',
    target_samples_per_class=None,
    seed=42
):
    """
    Balance the dataset by augmenting minority classes.
    
    Args:
        image_paths (list): List of paths to the original images
        labels (list): List of corresponding labels
        output_dir (str): Directory to store augmented images
        target_samples_per_class (int): Target number of samples per class. If None, uses max class count
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (augmented_image_paths, augmented_labels)
    """
    # Set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    
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
    
    # Define augmentation transforms
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])
    
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
            
            # Add original samples
            for idx in class_indices:
                augmented_paths.append(image_paths[idx])
                augmented_labels.append(labels[idx])
            
            # Generate augmented samples
            for i in range(samples_needed):
                # Randomly select an image from the current class
                source_idx = random.choice(class_indices)
                source_path = image_paths[source_idx]
                
                # Load and augment image
                image = Image.open(source_path)
                augmented_image = augmentation_transforms(image)
                
                # Generate unique filename
                base_name = Path(source_path).stem
                new_filename = f"{base_name}_aug_{i}.png"
                new_path = output_path / class_name / new_filename
                
                # Save augmented image
                augmented_image.save(new_path)
                
                # Add to augmented dataset
                augmented_paths.append(str(new_path))
                augmented_labels.append(class_name)
        else:
            # If class already has enough samples, just add original samples
            for idx in class_indices:
                augmented_paths.append(image_paths[idx])
                augmented_labels.append(labels[idx])
    
    return augmented_paths, augmented_labels

def merge_augmented_data(
    original_paths,
    original_labels,
    augmented_paths,
    augmented_labels,
    split_ratio=(0.7, 0.15, 0.15)
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
    df = pd.DataFrame({
        'image_path': all_paths,
        'label': all_labels
    })
    
    # Shuffle DataFrame
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split indices
    n_samples = len(df)
    train_size = int(n_samples * split_ratio[0])
    val_size = int(n_samples * split_ratio[1])
    
    # Split data
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    return train_df, val_df, test_df

def balance_dataset_with_augmentation(
    image_paths,
    labels,
    output_dir='new_aug',
    target_samples_per_class=None,
):
    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists. Please delete it or choose a different directory.")
        return

    # Apply data augmentation to balance the dataset
    print("Applying data augmentation to balance the dataset...")
    augmented_paths, augmented_labels = balance_dataset_with_augmentation(
        image_paths,
        labels,
        output_dir=output_dir,
        target_samples_per_class=target_samples_per_class
    )

    # Merge original and augmented data, then split into train/val/test
    train_df, val_df, test_df = merge_augmented_data(
        image_paths,
        labels,
        augmented_paths,
        augmented_labels,
        split_ratio=(0.7, 0.15, 0.15)
    )

    # Extract paths and labels from DataFrames
    train_image_paths = train_df['image_path'].tolist()
    train_labels = train_df['label'].tolist()
    val_image_paths = val_df['image_path'].tolist()
    val_labels = val_df['label'].tolist()
    test_image_paths = test_df['image_path'].tolist()
    test_labels = test_df['label'].tolist()

    return train_image_paths, train_labels, val_image_paths, val_labels, test_image_paths, test_labels