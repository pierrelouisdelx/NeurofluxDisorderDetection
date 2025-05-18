# Neuroflux Disorder Detection

A deep learning project for classifying different phases of Neuroflux disorder from medical images.

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
│   ├── datasets.py        # PyTorch Dataset implementation for loading and preprocessing medical images
│   ├── __init__.py        # Python package marker file
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

### Training Models

To train the transfer learning model (resnet50)

```bash
uv run src/main.py train --dataset_config configs/dataset_config.json --model_config configs/resnet50_config.json
```

To train the self made model

```bash
uv run src/main.py --dataset_config configs/dataset_config.json --model_config configs/neuroflux_config.json
```

### Evaluation

To evaluate the transfer learning model performance:

```bash
python src/main.py --mode evaluate --dataset_config configs/dataset_config.json --model_config configs/resnet50_config.json
```

To evaluate the self made model performance:

```bash
python src/main.py --mode evaluate --dataset_config configs/dataset_config.json --model_config configs/neuroflux_config.json
```

### Prediction

To predict the phase of a new MRI scan from the transfer learning model:

```bash
python src/main.py --mode predict --image path/to/image.jpg --dataset_config configs/dataset_config.json --model_config configs/resnet50_config.json
```

To predict the phase of a new MRI scan from the self made model:

```bash
python src/main.py --mode predict --image path/to/image.jpg --dataset_config configs/dataset_config.json --model_config configs/neuroflux_config.json
```

## Docker Usage

### Building the Docker Image

```bash
docker build -t neuroflux-detection .
```

### Running with Docker

```bash
docker run neuroflux-detection [train|evaluate|predict] --dataset_config [config_path] --model_config [model_config]
```

## Models

### Model 1: Transfer Learning (Resnet50)

-   Uses a pre-trained neural network architecture
-   Fine-tuned for the specific classification task
-   Implements transfer learning techniques

### Model 2: Custom Architecture

-   Implemented from scratch
-   Custom neural network architecture
-   Trained without pre-trained weights

## Performance Metrics

The models are evaluated using multiple metrics:

-   Accuracy
-   Precision
-   Recall
-   F1-Score
-   Confusion Matrix
