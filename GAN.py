import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
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

# Define the Generator
class Generator(nn.Module):
    def __init__(self, text_embedding_dim, noise_dim, image_size):
        super(Generator, self).__init__()
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

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, text_embedding_dim, image_size):
        super(Discriminator, self).__init__()
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

# Training Loop
def train_gan(image_folder, csv_path, epochs=100, batch_size=32, lr=0.0002, noise_dim=100, image_size=64):
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
    generator = Generator(text_embedding_dim=512, noise_dim=noise_dim, image_size=image_size).to(device)
    discriminator = Discriminator(text_embedding_dim=512, image_size=image_size).to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

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

            # Train Discriminator
            optimizer_D.zero_grad()

            real_outputs = discriminator(images, text_embeddings)
            real_loss = adversarial_loss(real_outputs, real_labels)

            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_images = generator(noise, text_embeddings)
            fake_outputs = discriminator(fake_images.detach(), text_embeddings)
            fake_loss = adversarial_loss(fake_outputs, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()

            fake_outputs = discriminator(fake_images, text_embeddings)
            g_loss = adversarial_loss(fake_outputs, real_labels)

            g_loss.backward()
            optimizer_G.step()

            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Save the generator model
    torch.save(generator.state_dict(), "generator.pth")

# Generate Image
def generate_image(generator_path, text, noise_dim=100, image_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    generator = Generator(text_embedding_dim=512, noise_dim=noise_dim, image_size=image_size).to(device)
    generator.load_state_dict(torch.load(generator_path))
    generator.eval()

    text_tokens = tokenizer([text], padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
    text_embeddings = text_model(text_tokens).pooler_output

    noise = torch.randn(1, noise_dim).to(device)
    with torch.no_grad():
        fake_image = generator(noise, text_embeddings).cpu().squeeze()

    fake_image = fake_image.permute(1, 2, 0) * 0.5 + 0.5  # Denormalize
    fake_image = (fake_image.numpy() * 255).astype(np.uint8)
    fake_image = Image.fromarray(fake_image)
    fake_image.save("generated_image.png")

# Example Usage
# train_gan("path_to_images", "path_to_csv", epochs=50)
# generate_image("generator.pth", "A description of the image")
