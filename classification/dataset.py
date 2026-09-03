import os
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError


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
                        # Attempt to open the image to ensure it's valid
                        with Image.open(img_path) as img:
                            img.verify()  # Verify if it's an actual image
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
