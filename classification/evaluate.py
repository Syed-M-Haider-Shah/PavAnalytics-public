import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb

from .config import device


# Evaluation function
def evaluate_model(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total_val += labels.size(0)
            correct_val += predicted.eq(labels).sum().item()

            torch.cuda.empty_cache()

    val_accuracy = 100 * correct_val / total_val
    val_loss_avg = val_loss / len(loader)
    return val_loss_avg, val_accuracy


# Test the model
def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    class_names = ["1", "2", "3", "4", "5"]

    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nTest Set Classification Report:\n", report)

    cm = confusion_matrix(all_labels, all_preds)

    test_accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    test_f1_score = f1_score(all_labels, all_preds, average='weighted')
    print(f"Test F1 Score: {test_f1_score:.4f}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    wandb.log({
        'test_accuracy': test_accuracy,
        'test_f1_score': test_f1_score
    })
