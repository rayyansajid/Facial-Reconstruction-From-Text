import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

# Define Dataset Class
class TextImageDataset(Dataset):
    def __init__(self, image_folder, csv_path, transform=None):
        self.image_folder = image_folder
        self.csv_data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        image_name = self.csv_data.iloc[idx, 0]
        description = self.csv_data.iloc[idx, 1]

        # Load image
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, description

# Define Stage-I Generator
class Stage1Generator(nn.Module):
    def __init__(self, text_embedding_dim, noise_dim, image_size):
        super(Stage1Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_embedding_dim + noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, image_size * image_size * 3),
            nn.Tanh()
        )
        self.image_size = image_size

    def forward(self, noise, text_embedding):
        x = torch.cat((noise, text_embedding), dim=1)
        x = self.fc(x)
        x = x.view(-1, 3, self.image_size, self.image_size)
        return x

# Define Stage-I Discriminator
class Stage1Discriminator(nn.Module):
    def __init__(self, text_embedding_dim, image_size):
        super(Stage1Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3 * image_size * image_size + text_embedding_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.image_size = image_size

    def forward(self, image, text_embedding):
        image_flat = image.view(image.size(0), -1)
        x = torch.cat((image_flat, text_embedding), dim=1)
        x = self.fc(x)
        return x

# Define Stage-II Generator
class Stage2Generator(nn.Module):
    def __init__(self, text_embedding_dim, stage1_image_channels, image_size):
        super(Stage2Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(stage1_image_channels + text_embedding_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.image_size = image_size

    def forward(self, stage1_image, text_embedding):
        text_embedding = text_embedding.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.image_size, self.image_size)
        x = torch.cat((stage1_image, text_embedding), dim=1)
        x = self.fc(x)
        return x

# Define Stage-II Discriminator
class Stage2Discriminator(nn.Module):
    def __init__(self, text_embedding_dim, image_size):
        super(Stage2Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3 * image_size * image_size + text_embedding_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.image_size = image_size

    def forward(self, image, text_embedding):
        image_flat = image.view(image.size(0), -1)
        x = torch.cat((image_flat, text_embedding), dim=1)
        x = self.fc(x)
        return x

# Training Loop for StackGAN
def train_stack_gan(image_folder, csv_path, epochs=100, batch_size=32, lr=0.0002, noise_dim=100, image_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = TextImageDataset(image_folder, csv_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load text model
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    # Initialize models
    stage1_generator = Stage1Generator(text_embedding_dim=512, noise_dim=noise_dim, image_size=image_size).to(device)
    stage1_discriminator = Stage1Discriminator(text_embedding_dim=512, image_size=image_size).to(device)

    stage2_generator = Stage2Generator(text_embedding_dim=512, stage1_image_channels=3, image_size=image_size).to(device)
    stage2_discriminator = Stage2Discriminator(text_embedding_dim=512, image_size=image_size).to(device)

    # Optimizers
    optimizer_G1 = optim.Adam(stage1_generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D1 = optim.Adam(stage1_discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    optimizer_G2 = optim.Adam(stage2_generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D2 = optim.Adam(stage2_discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss function
    adversarial_loss = nn.BCELoss().to(device)

    # Training
    for epoch in range(epochs):
        for i, (images, descriptions) in enumerate(dataloader):
            batch_size = images.size(0)

            # Prepare real and fake labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Prepare images and text embeddings
            images = images.to(device)
            text_tokens = tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
            text_embeddings = text_model(text_tokens).pooler_output

            # Train Stage-I Discriminator
            optimizer_D1.zero_grad()

            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_images_stage1 = stage1_generator(noise, text_embeddings)

            real_outputs_stage1 = stage1_discriminator(images, text_embeddings)
            real_loss_stage1 = adversarial_loss(real_outputs_stage1, real_labels)

            fake_outputs_stage1 = stage1_discriminator(fake_images_stage1.detach(), text_embeddings)
            fake_loss_stage1 = adversarial_loss(fake_outputs_stage1, fake_labels)

            d1_loss = real_loss_stage1 + fake_loss_stage1
            d1_loss.backward()
            optimizer_D1.step()

            # Train Stage-I Generator
            optimizer_G1.zero_grad()

            fake_outputs_stage1 = stage1_discriminator(fake_images_stage1, text_embeddings)
            g1_loss = adversarial_loss(fake_outputs_stage1, real_labels)

            g1_loss.backward()
            optimizer_G1.step()

            # Train Stage-II Discriminator
            optimizer_D2.zero_grad()

            fake_images_stage2 = stage2_generator(fake_images_stage1, text_embeddings)

            real_outputs_stage2 = stage2_discriminator(images, text_embeddings)
            real_loss_stage2 = adversarial_loss(real_outputs_stage2, real_labels)

            fake_outputs_stage2 = stage2_discriminator(fake_images_stage2.detach(), text_embeddings)
            fake_loss_stage2 = adversarial_loss(fake_outputs_stage2, fake_labels)

            d2_loss = real_loss_stage2 + fake_loss_stage2
            d2_loss.backward()
            optimizer_D2.step
