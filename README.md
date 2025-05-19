# Neuroflux Disorder Detection

A deep learning project for classifying different phases of Neuroflux disorder from medical images. The disease is an imaginary disease that has been created for this project.

## Overview

This project implements two deep learning models to classify medical images into five different phases of Neuroflux disorder:

-   EO: Early Onset Neuroflux disorder
-   IO: Intermediate onset Neuroflux disorder
-   LO: Late Onset Neuroflux disorder
-   PTE: Neuroflux disorder with polyglutamine tract expansion
-   IPTE: Neuroflux disorder with intermediate polyglutamine tract expansion

## Project Structure

```
.
├── data/                  # Dataset directory (not included in repository)
│   ├── EO/                # Early Onset Neuroflux disorder images
│   ├── IO/                # Intermediate onset Neuroflux disorder images
│   ├── LO/                # late Onset Neuroflux disorder images
│   ├── PTE/               # Neuroflux disorder with polyglutamine tract expansion images
│   └── IPTE/              # Neuroflux disorder with intermediate polyglutamine tract expansion images
├── README.md
├── src
│   ├── dataset.py        # PyTorch Dataset implementation for loading and preprocessing medical images
│   ├── hyperparameter_tuning.py  # Hyperparameter tuning using optuna
│   ├── __init__.py       # Python package marker file
│   ├── main.py           # Entry point of the application, handles CLI and orchestrates workflows
│   ├── models.py         # Neural network architectures (ResNet50 and custom model)
│   ├── training.py       # Training loop implementation with validation and early stopping
│   └── utils
│       ├── config_loader.py      # JSON configuration file loader
│       ├── data_augmentation.py  # Data augmentation and dataset balancing utilities
│       ├── file_utils.py         # File operations and data splitting utilities
│       ├── __init__.py           # Python package marker file
│       ├── preprocessing.py      # MRI image preprocessing and quality improvement
│       └── transforms.py         # Image transformation pipelines for training and validation
├── Dockerfile            # Instructions for building the Docker container
├── uv.lock              # Dependency lock file for uv package manager
└── pyproject.toml       # Project configuration and dependencies
```

## Requirements

-   Python 3.8+
-   PyTorch
-   CUDA (for GPU acceleration)
-   Docker (for containerization)
-   Nvidia Container Toolkit (for GPU acceleration in Docker)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/pierrelouisdelx/NeurofluxDisorderDetection.git
cd NeurofluxDisorderDetection
```

2. Install dependencies:

In order to install the dependencies uv is required. If you do not have uv installed, please refer to the [installation guide](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync
```

## Usage

The main entry point of the application is located in `src/main.py` and handles different modes of operation for the Neuroflux Disorder Phase Classifier.

Modes:

-   `preprocess`: Analyze and preprocess the dataset
-   `train`: Train the model
-   `evaluate`: Evaluate the model on test data
-   `predict`: Make predictions on a single image

### Training Models

The dataset configuration file is located in the `configs` folder. It has a very simple structure:

```json
{
    "data_dir": "data", # Path to the dataset
    "image_size": [224, 224], # Size of the images
    "train_val_test_split": [0.7, 0.15, 0.15], # Split of the dataset into train, validation and test
    "num_workers": 4, # Number of workers for the dataloader
    "random_seed": 42, # Random seed for the dataset
    "class_names": ["PTE", "LO", "EO", "IPTE", "IO"] # Names of the classes
}
```

Each model has its own configuration file located in the `configs` folder. The configuration files are named as follows:

-   `resnet50_config.json`: Configuration for the ResNet50 model

    ```json
    {
        "model_name": "resnet50",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "num_epochs": 50,
        "model_save_path": "saved_models/resnet50_model.pth",
        "batch_size": 64
    }
    ```

-   `neuroflux_config.json`: Configuration for the custom model

    ```json
    {
        "model_name": "neuroflux",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "num_epochs": 100,
        "model_save_path": "saved_models/neuroflux_model.pth",
        "batch_size": 32
    }
    ```

To train the model

```bash
uv run src/main.py train --dataset_config configs/dataset_config.json --model_config configs/<model_name>_config.json
```

#### Training Progress

The training progress is logged in the `output/runs` directory. To monitor the training progress, the tensorboard command can be used:

```bash
tensorboard --logdir=output/runs
```

### Evaluation

To evaluate the model performances:

```bash
python src/main.py --mode evaluate --dataset_config configs/dataset_config.json --model_config configs/<model_name>_config.json
```

### Prediction

To predict the phase of a new MRI scan:

```bash
python src/main.py --mode predict --image path/to/image.jpg --dataset_config configs/dataset_config.json --model_config configs/<model_name>_config.json
```

## Docker Usage

In order to be able to use the models with Docker, nvidia-container-toolkit is required. Please refer to the [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for more information.

### Building the Docker Image

```bash
docker build -t neuroflux-detection .
```

### Running with Docker

```bash
docker run neuroflux-detection [train|evaluate|predict] --dataset_config configs/dataset_config.json --model_config <model_config>
```

# Data Preparation

## Initial Observations

The dataset provided was highly imbalanced and contained many low-quality images. The first step was to clean the dataset by removing these low-quality images.

## Data Augmentation

To address the imbalance in the dataset, a data augmentation script was developed. This script generates an equal number of images for each class using the following transformations:

```python
self.augmentation_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=(-15, 15), scale=(0.9, 1.1)),
    ]
)
```

## Model Development

Model 1: Transfer Learning with ResNet50
A pre-trained ResNet50 model was fine-tuned to classify the phases of Neuroflux disorder. This approach leverages transfer learning to improve the model's performance.

Model 2: Custom CNN
A custom Convolutional Neural Network (CNN) was designed and trained from scratch. The architecture of the custom model is as follows:

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
NeurofluxModel                           [1, 5]                    --
├─Conv2d: 1-1                            [1, 32, 224, 224]         896
├─MaxPool2d: 1-2                         [1, 32, 112, 112]         --
├─Conv2d: 1-3                            [1, 64, 112, 112]         18,496
├─MaxPool2d: 1-4                         [1, 64, 56, 56]           --
├─Conv2d: 1-5                            [1, 128, 56, 56]          73,856
├─MaxPool2d: 1-6                         [1, 128, 28, 28]          --
├─Conv2d: 1-7                            [1, 256, 28, 28]          295,168
├─MaxPool2d: 1-8                         [1, 256, 14, 14]          --
├─Flatten: 1-9                           [1, 50176]                --
├─Linear: 1-10                           [1, 256]                  12,845,312
├─Dropout: 1-11                          [1, 256]                  --
├─Linear: 1-12                           [1, 5]                    1,285
==========================================================================================
Total params: 13,235,013
Trainable params: 13,235,013
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 752.84
==========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 24.09
Params size (MB): 52.94
Estimated Total Size (MB): 77.63
==========================================================================================
```

## Hyperparameter Tuning

Optuna was used for hyperparameter tuning to optimize the performance of both models. Training progress was monitored using TensorBoard. All training and hyperparameter tuning were done on a T4 GPU on Kaggle and google colab.

## Performance Metrics

The models are evaluated using multiple metrics:

-   Accuracy
-   Precision
-   Recall
-   F1-Score
-   Confusion Matrix

## Results

### Model 1: Transfer Learning with ResNet50

### Model 2: Custom CNN
