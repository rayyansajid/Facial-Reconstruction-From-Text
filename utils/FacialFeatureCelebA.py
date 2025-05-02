import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

IMAGE_FOLDER_PATH = os.getenv("RESNET18_IMG_FOLDER")
ATTR_FILE = os.getenv("RESNET18_ATTR_FILE")
PARTITION_FILE = os.getenv("RESNET18_PARTITION_FILE")

# Dataset Class
class CelebADataset(Dataset):
    def __init__(self, root_dir, attr_file, partition_file, partition, transform=None):
        self.root_dir = root_dir
        self.attr_data = pd.read_csv(attr_file)
        self.partition_data = pd.read_csv(partition_file)
        
        # Filter images based on the partition (0: Train, 1: Val, 2: Test)
        self.images = self.partition_data[self.partition_data['partition'] == partition]['image_id'].values
        self.labels = self.attr_data[self.attr_data['image_id'].isin(self.images)].iloc[:, 1:].values

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")
        label = torch.tensor((self.labels[idx] + 1) // 2, dtype=torch.float32)  # Convert {-1, 1} to {0, 1}

        if self.transform:
            image = self.transform(image)

        return image, label


# Function for training and testing
def train_model():
    data_dir = rf"{IMAGE_FOLDER_PATH}"  # Folder containing images
    attr_file = rf"{ATTR_FILE}"
    partition_file = rf"{PARTITION_FILE}"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    batch_size = 128
    num_workers = 4

    train_dataset = CelebADataset(data_dir, attr_file, partition_file, partition=0, transform=transform)
    val_dataset = CelebADataset(data_dir, attr_file, partition_file, partition=1, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model (Pretrained ResNet18)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)  # Force GPU 0
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 40)  # Output 40 attributes
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader)}")

    # Save the Model
    model_path = "celeba_model.pth"
    torch.save(model.state_dict(), model_path)
    print("Model saved!")


# Windows-Specific Multiprocessing Protection
if __name__ == '__main__':
    # This is required for Windows to avoid multiprocessing errors
    torch.multiprocessing.set_start_method('spawn', force=True)
    train_model()