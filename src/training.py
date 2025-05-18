import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from torch.utils.tensorboard import SummaryWriter

OUTPUT_DIR = 'output'

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_save_path, class_names, writer=None):
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
    
    # Log model graph
    if writer is not None:
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        writer.add_graph(model, dummy_input)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
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
            
            # Log batch metrics
            if writer is not None and batch_idx % 5 == 0:
                writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)
                batch_acc = 100. * (predicted == labels).sum().item() / labels.size(0)
                writer.add_scalar('Accuracy/train_batch', batch_acc, epoch * len(train_loader) + batch_idx)
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
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
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if scheduler:
            scheduler.step(val_loss)
            if writer is not None:
                writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log epoch metrics
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 50)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Plot and save training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'))
    plt.close()
    
    return model

def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluate model on test set with confusion matrix and classification report
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.show()
    
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return all_preds, all_labels