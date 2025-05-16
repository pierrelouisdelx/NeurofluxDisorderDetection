import argparse
import torch
from torch.utils.data import DataLoader
import os
import random
import numpy as np

from neuroflux_analyzer.utils.config_loader import load_config

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

if __name__ == '__main__':
    main()