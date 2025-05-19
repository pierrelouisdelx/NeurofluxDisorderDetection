import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report, recall_score, roc_curve, auc, f1_score, precision_score
from sklearn.preprocessing import label_binarize
import numpy as np

OUTPUT_DIR = "output"


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
    model_save_path,
    class_names,
    writer=None,
):
    """
    Training function with validation and early stopping
    """
    best_val_acc = 0.0
    patience = 5
    counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_f1s = []
    val_f1s = []
    train_precisions = []
    val_precisions = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Lists to store predictions and labels for metrics calculation
        train_preds = []
        train_labels = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions and labels for metrics calculation
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            # Log batch metrics
            if writer is not None and batch_idx % 5 == 0:
                writer.add_scalar(
                    "Loss/train_batch",
                    loss.item(),
                    epoch * len(train_loader) + batch_idx,
                )
                batch_acc = 100.0 * (predicted == labels).sum().item() / labels.size(0)
                writer.add_scalar(
                    "Accuracy/train_batch",
                    batch_acc,
                    epoch * len(train_loader) + batch_idx,
                )

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # Calculate metrics for training set
        train_recall = recall_score(train_labels, train_preds, average=None)
        train_f1 = f1_score(train_labels, train_preds, average=None)
        train_precision = precision_score(train_labels, train_preds, average=None, zero_division=0)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        train_precisions.append(train_precision)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Lists to store predictions and labels for metrics calculation
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Store predictions and labels for metrics calculation
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        # Calculate metrics for validation set
        val_recall = recall_score(val_labels, val_preds, average=None)
        val_f1 = f1_score(val_labels, val_preds, average=None)
        val_precision = precision_score(val_labels, val_preds, average=None, zero_division=0)

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_precisions.append(val_precision)

        if scheduler:
            scheduler.step()
            if writer is not None:
                writer.add_scalar(
                    "Learning_rate", optimizer.param_groups[0]["lr"], epoch
                )

        # Log epoch metrics
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

            # Log recall metrics for each class
            for i, class_name in enumerate(class_names):
                writer.add_scalar(f"Recall/train_{class_name}", train_recall[i], epoch)
                writer.add_scalar(f"Recall/val_{class_name}", val_recall[i], epoch)
                writer.add_scalar(f"F1/train_{class_name}", train_f1[i], epoch)
                writer.add_scalar(f"F1/val_{class_name}", val_f1[i], epoch)
                writer.add_scalar(f"Precision/train_{class_name}", train_precision[i], epoch)
                writer.add_scalar(f"Precision/val_{class_name}", val_precision[i], epoch)

            # Log average metrics
            writer.add_scalar("Recall/train_avg", train_recall.mean(), epoch)
            writer.add_scalar("Recall/val_avg", val_recall.mean(), epoch)
            writer.add_scalar("F1/train_avg", train_f1.mean(), epoch)
            writer.add_scalar("F1/val_avg", val_f1.mean(), epoch)
            writer.add_scalar("Precision/train_avg", train_precision.mean(), epoch)
            writer.add_scalar("Precision/val_avg", val_precision.mean(), epoch)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 50)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            model_save_path_with_acc = (
                f"{model_save_path.rsplit('.', 1)[0]}_acc_{val_acc:.2f}.pth"
            )
            torch.save(model.state_dict(), model_save_path_with_acc)
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Plot and save training curves
    plt.figure(figsize=(15, 10))

    # Plot 1: Loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot 2: Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # Plot 3: F1 Score
    plt.subplot(2, 2, 3)
    for i, class_name in enumerate(class_names):
        train_f1_class = [f1[i] for f1 in train_f1s]
        val_f1_class = [f1[i] for f1 in val_f1s]
        plt.plot(train_f1_class, label=f"Train {class_name}")
        plt.plot(val_f1_class, label=f"Val {class_name}")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()

    # Plot 4: Precision
    plt.subplot(2, 2, 4)
    for i, class_name in enumerate(class_names):
        train_precision_class = [p[i] for p in train_precisions]
        val_precision_class = [p[i] for p in val_precisions]
        plt.plot(train_precision_class, label=f"Train {class_name}")
        plt.plot(val_precision_class, label=f"Val {class_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"))
    plt.close()

    return model


def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluate model on test set with confusion matrix, classification report, and ROC curves
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Binarize the labels for ROC curve calculation
    n_classes = len(class_names)
    y_test_bin = label_binarize(all_labels, classes=range(n_classes))

    # Calculate ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve of {class_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"))
    plt.close()

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    print(classification_report(all_labels, all_preds, target_names=class_names))

    return all_preds, all_labels
