# This model results are to be put in confrence paper.
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForImageClassification, get_scheduler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import wandb
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm  # Progress bar

# Initialize Weights & Biases
wandb.init(project='image-classification', config={
    "learning_rate": 2e-5,
    "epochs": 50,
    "batch_size": 64,
    "model_name": "microsoft/swin-tiny-patch4-window7-224"
})

config = wandb.config

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        class_names = ["1", "2", "3", "4", "5"]  # Update class names for 5 classes

        for class_idx, class_label in enumerate(class_names):
            class_path = os.path.join(root_dir, class_label)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)
                    except (UnidentifiedImageError, OSError):
                        print(f"Skipping non-image file: {img_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

data_transforms_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), hue=(-0.1, 0.1)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_transforms_val_test = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomImageDataset(root_dir=r"D:\distributed_data_70_15_15-new\Train", transform=data_transforms_train)
val_dataset = CustomImageDataset(root_dir=r"D:\distributed_data_70_15_15-new\Validation", transform=data_transforms_val_test)
test_dataset = CustomImageDataset(root_dir=r"D:\distributed_data_70_15_15-new\Test", transform=data_transforms_val_test)

print(f"Training dataset size: {len(train_dataset)} images")
print(f"Validation dataset size: {len(val_dataset)} images")
print(f"Testing dataset size: {len(test_dataset)} images")

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

model_name = config.model_name
model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=5, ignore_mismatched_sizes=True)
model.to(device)

checkpoint_file = r'C:\Users\Syed\Swim-Transformer\checkpoints\all_model_checkpoints_2layer_2e-5_all-road-surface dataset-2nd.pth'
checkpoint = torch.load(checkpoint_file)
filtered_checkpoint = {k: v for k, v in checkpoint['epoch_50'].items() if not k.startswith("classifier")}
model.load_state_dict(filtered_checkpoint, strict=False)

mismatches = 0
for name, param in model.named_parameters():
    if name in filtered_checkpoint:
        if not torch.equal(param.data, filtered_checkpoint[name]):
            print(f"Mismatch found in layer: {name}")
            mismatches += 1

if mismatches == 0:
    print("All non-classifier layers are loaded correctly from the checkpoint.")
else:
    print(f"{mismatches} layers have mismatched weights compared to the checkpoint.")

for param in model.swin.encoder.layers[:-2].parameters():
    param.requires_grad = False

optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * config.epochs)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()
results = []

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_file):
    checkpoints = {}
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}/{num_epochs}')
        model.train(); running_loss = 0.0; correct_train = 0; total_train = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images).logits
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
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
        results.append({'epoch': epoch + 1,'train_loss': train_loss_avg,'train_accuracy': train_accuracy,'val_loss': val_loss,'val_accuracy': val_acc})
        print(f'Epoch {epoch+1} - Train Loss: {train_loss_avg:.4f} - Train Accuracy: {train_accuracy:.2f}% - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.2f}%')
        wandb.log({'epoch': epoch + 1,'train_loss': train_loss_avg,'train_accuracy': train_accuracy,'val_loss': val_loss,'val_accuracy': val_acc})
    torch.save(checkpoints, checkpoint_file)
    print(f"All checkpoints saved at {checkpoint_file}")
    print('Training complete.')

def evaluate_model(model, loader, criterion):
    model.eval(); val_loss = 0.0; correct_val = 0; total_val = 0
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
    return val_loss / len(loader), 100 * correct_val / total_val

checkpoint_file = r'./checkpoints/all_model_checkpoints_2layer_2e-5_retrained_all-road-surface dataset-2nd.pth'
os.makedirs('./checkpoints', exist_ok=True)
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config.epochs, checkpoint_file)
results_df = pd.DataFrame(results)
results_df.to_csv(r'training_results_2layer_2e-5_retrain-with-spili-1.csv', index=False)
print("Training results saved to training_results_2layer_2e-5_retrain-with-spili-1.csv")

def test_model(model, test_loader):
    model.eval(); all_preds = []; all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
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
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix'); plt.show()
    wandb.log({'test_accuracy': test_accuracy,'test_f1_score': test_f1_score})

test_model(model, test_loader)
