import os
from glob import glob
import random
import pandas as pd

def load_outlier_paths_csv(outlier_path):
    if os.path.exists(outlier_path):
        outliers = pd.read_csv(outlier_path)
        return outliers['path'].tolist()
    else:
        return []

def get_label(image_path):
    return os.path.basename(os.path.dirname(image_path))

def get_images_and_labels(data_dir):
    exclude_paths = load_outlier_paths_csv('dataset_analysis/outliers.csv')
    print(f"Excluding {len(exclude_paths)} outliers")

    image_paths = []
    labels = []

    all_image_paths = glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True)
    for image_path in all_image_paths:
        if image_path in exclude_paths:
            continue

        label = get_label(image_path)
        image_paths.append(image_path)
        labels.append(label)

    return image_paths, labels

def split_data(image_paths, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Group paths by label
    label_groups = {}
    for path, label in zip(image_paths, labels):
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(path)
    
    train_image_paths = []
    train_labels = []
    val_image_paths = []
    val_labels = []
    test_image_paths = []
    test_labels = []
    
    # Split each label group proportionally
    for label, paths in label_groups.items():
        random.shuffle(paths)
        
        n_samples = len(paths)
        train_size = int(train_ratio * n_samples)
        val_size = int(val_ratio * n_samples)
        
        train_paths = paths[:train_size]
        val_paths = paths[train_size:train_size + val_size]
        test_paths = paths[train_size + val_size:]
        
        train_image_paths.extend(train_paths)
        train_labels.extend([label] * len(train_paths))
        val_image_paths.extend(val_paths)
        val_labels.extend([label] * len(val_paths))
        test_image_paths.extend(test_paths)
        test_labels.extend([label] * len(test_paths))
    
    return train_image_paths, train_labels, val_image_paths, val_labels, test_image_paths, test_labels

    



