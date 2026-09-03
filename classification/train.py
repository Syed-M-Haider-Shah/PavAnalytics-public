import torch
import wandb
from tqdm import tqdm  # Progress bar

from .config import device
from .evaluate import evaluate_model


# Training function to save checkpoints in a single file
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_file, scaler, results):
    checkpoints = {}  # Dictionary to store model state for each epoch

    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}/{num_epochs}')
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images).logits
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

            torch.cuda.empty_cache()

        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        scheduler.step()

        train_accuracy = 100 * correct_train / total_train
        train_loss_avg = running_loss / len(train_loader)

        checkpoints[f'epoch_{epoch+1}'] = model.state_dict()

        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss_avg,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })

        print(f'Epoch {epoch+1} - '
              f'Train Loss: {train_loss_avg:.4f} - Train Accuracy: {train_accuracy:.2f}% - '
              f'Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.2f}%')

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss_avg,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })

    torch.save(checkpoints, checkpoint_file)
    print(f"All checkpoints saved at {checkpoint_file}")

    print('Training complete.')
