import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from dataset import NeurofluxDataset
from models.model_factory import ModelFactory
from training import train_model, evaluate_model
from utils.transforms import get_train_transforms, get_val_test_transforms
from utils.file_utils import get_images_and_labels
from utils.data_augmenter import DataAugmenter
from utils.config_loader import load_config

def get_optimizer(trial, model_parameters):
    """
    Get optimizer based on trial suggestions
    """
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD", "RMSprop"])
    
    if optimizer_name == "Adam":
        return optim.Adam(
            model_parameters,
            lr=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            betas=(
                trial.suggest_float("beta1", 0.8, 0.99),
                trial.suggest_float("beta2", 0.9, 0.9999)
            )
        )
    elif optimizer_name == "AdamW":
        return optim.AdamW(
            model_parameters,
            lr=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            betas=(
                trial.suggest_float("beta1", 0.8, 0.99),
                trial.suggest_float("beta2", 0.9, 0.9999)
            )
        )
    elif optimizer_name == "SGD":
        return optim.SGD(
            model_parameters,
            lr=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            momentum=trial.suggest_float("momentum", 0.8, 0.99),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            nesterov=trial.suggest_categorical("nesterov", [True, False])
        )
    else:  # RMSprop
        return optim.RMSprop(
            model_parameters,
            lr=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            momentum=trial.suggest_float("momentum", 0.8, 0.99),
            alpha=trial.suggest_float("alpha", 0.8, 0.99)
        )

def get_scheduler(trial, optimizer):
    """
    Get learning rate scheduler based on trial suggestions
    """
    scheduler_name = trial.suggest_categorical("scheduler", ["ReduceLROnPlateau", "CosineAnnealingLR", "OneCycleLR"])
    
    if scheduler_name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=trial.suggest_float("lr_factor", 0.1, 0.5),
            patience=trial.suggest_int("lr_patience", 3, 10),
            min_lr=trial.suggest_float("min_lr", 1e-6, 1e-4, log=True)
        )
    elif scheduler_name == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=trial.suggest_int("T_max", 10, 100),
            eta_min=trial.suggest_float("eta_min", 1e-6, 1e-4, log=True)
        )
    else:  # OneCycleLR
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=trial.suggest_float("max_lr", 1e-4, 1e-2, log=True),
            total_steps=trial.suggest_int("total_steps", 100, 1000),
            pct_start=trial.suggest_float("pct_start", 0.1, 0.3),
            div_factor=trial.suggest_float("div_factor", 10, 100),
            final_div_factor=trial.suggest_float("final_div_factor", 100, 1000)
        )

def objective(trial, model_name, train_loader, val_loader, test_loader, device, class_names, output_dir):
    """
    Optuna objective function for hyperparameter optimization
    """
    # Define hyperparameter search space
    if model_name == "resnet50":
        hyperparams = {
            "dropout_rate": trial.suggest_float("dropout_rate", 0.3, 0.7),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "hidden_size": trial.suggest_categorical("hidden_size", [256, 512, 1024]),
        }
    else:  # neuroflux
        hyperparams = {
            "dropout_rate": trial.suggest_float("dropout_rate", 0.3, 0.7),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "conv_channels": trial.suggest_categorical("conv_channels", [16, 32, 64]),
        }

    # Create model with trial hyperparameters
    model = ModelFactory(model_name, len(class_names), hyperparams=hyperparams).get_model()
    model.to(device)

    # Setup optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(trial, model.parameters())
    scheduler = get_scheduler(trial, optimizer)

    # Setup TensorBoard
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(output_dir, "runs", f"{model_name}_trial_{trial.number}")
    writer = SummaryWriter(log_dir=log_dir)

    # Train model
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,  # Fixed number of epochs for hyperparameter tuning
        device=device,
        model_save_path=os.path.join(output_dir, f"{model_name}_trial_{trial.number}.pth"),
        class_names=class_names,
        writer=writer
    )

    # Evaluate on validation set
    val_preds, val_labels = evaluate_model(model, val_loader, device, class_names)
    
    # Calculate validation accuracy
    val_acc = (val_preds == val_labels).mean()

    # Evaluate on test set
    test_preds, test_labels = evaluate_model(model, test_loader, device, class_names)
    
    # Calculate test accuracy
    test_acc = (test_preds == test_labels).mean()
    
    # Log hyperparameters and metrics
    trial.set_user_attr("val_accuracy", val_acc)
    trial.set_user_attr("test_accuracy", test_acc)
    
    return val_acc

def optimize_hyperparameters(model_name, dataset_config_path, output_dir="output/hyperparameter_tuning"):
    """
    Main function to optimize hyperparameters for a given model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset configuration
    dataset_cfg = load_config(dataset_config_path)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    image_paths, labels = get_images_and_labels(dataset_cfg.get("data_dir"))
    train_image_paths, train_labels, val_image_paths, val_labels, test_image_paths, test_labels = DataAugmenter().process_and_balance_dataset(
        image_paths, labels
    )
    
    # Create datasets
    train_dataset = NeurofluxDataset(
        train_image_paths,
        train_labels,
        transform=get_train_transforms(dataset_cfg.get("image_size")),
        class_names=dataset_cfg.get("class_names")
    )
    val_dataset = NeurofluxDataset(
        val_image_paths,
        val_labels,
        transform=get_val_test_transforms(dataset_cfg.get("image_size")),
        class_names=dataset_cfg.get("class_names")
    )
    test_dataset = NeurofluxDataset(
        test_image_paths,
        test_labels,
        transform=get_val_test_transforms(dataset_cfg.get("image_size")),
        class_names=dataset_cfg.get("class_names")
    )
    
    # Create data loaders with initial batch size (will be updated in trials)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model_name}_optimization",
        storage=f"sqlite:///{output_dir}/{model_name}_study.db",
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial,
            model_name,
            train_loader,
            val_loader,
            test_loader,
            device,
            dataset_cfg.get("class_names"),
            output_dir
        ),
        n_trials=50,  # Number of trials to run
        timeout=3600  # Timeout in seconds (1 hour)
    )
    
    # Print best results
    print(f"\nBest trial for {model_name}:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best hyperparameters
    best_params = {
        "model_name": model_name,
        "best_params": study.best_trial.params,
        "best_value": study.best_trial.value
    }
    
    import json
    with open(os.path.join(output_dir, f"{model_name}_best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)
    
    return study.best_trial.params

if __name__ == "__main__":
    # Optimize hyperparameters for both models
    dataset_config_path = "configs/dataset_config.json"
    
    print("Optimizing ResNet50 hyperparameters...")
    resnet50_params = optimize_hyperparameters("resnet50", dataset_config_path)
    
    print("\nOptimizing Neuroflux hyperparameters...")
    neuroflux_params = optimize_hyperparameters("neuroflux", dataset_config_path)