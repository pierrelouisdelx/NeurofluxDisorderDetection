import argparse
import torch
import random
import numpy as np
from torch.utils.data import DataLoader

from neuroflux_analyzer.utils.config_loader import load_config
from neuroflux_analyzer.utils.file_utils import get_images_and_labels, split_data
from neuroflux_analyzer.models import get_model
from neuroflux_analyzer.datasets import NeurofluxDataset
from neuroflux_analyzer.utils.transforms import get_train_transforms, get_val_test_transforms

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
    parser.add_argument('mode', choices=['train', 'evaluate', 'predict'], help="Mode to run: 'train', 'evaluate', or 'predict'")
    parser.add_argument('--dataset_config', type=str, default='configs/dataset_config.json', help="Path to the dataset configuration JSON file")
    parser.add_argument('--model_config', type=str, default='configs/model_config.json', help="Path to the model configuration JSON file")
    parser.add_argument('--image_path', type=str, help="Path to the MRI scan image for prediction")

    args = parser.parse_args()

    dataset_cfg = load_config(args.dataset_config)
    model_cfg = load_config(args.model_config)

    set_seed(dataset_cfg.get('random_seed', 42)) # Set seed for reproducibility

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    image_paths, labels = get_images_and_labels(dataset_cfg.get('data_dir'))
    train_image_paths, train_labels, val_image_paths, val_labels, test_image_paths, test_labels = split_data(image_paths, labels)

    print(f"Train set: {len(train_image_paths)} images")
    print(f"Validation set: {len(val_image_paths)} images")
    print(f"Test set: {len(test_image_paths)} images")

    # Load data
    train_dataset = NeurofluxDataset(train_image_paths, train_labels, transform=get_train_transforms(dataset_cfg.get('image_size')))
    val_dataset = NeurofluxDataset(val_image_paths, val_labels, transform=get_val_test_transforms(dataset_cfg.get('image_size')))
    test_dataset = NeurofluxDataset(test_image_paths, test_labels, transform=get_val_test_transforms(dataset_cfg.get('image_size')))

    train_loader = DataLoader(train_dataset, batch_size=dataset_cfg.get('batch_size'), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=dataset_cfg.get('batch_size'), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=dataset_cfg.get('batch_size'), shuffle=False)

    # Load model
    model = get_model(model_cfg.get('model_name'), len(dataset_cfg.get('classes')))
    model.to(device)



if __name__ == '__main__':
    main()