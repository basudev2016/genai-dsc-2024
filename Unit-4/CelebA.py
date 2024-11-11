import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Set up
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory for saving generated images
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# Generator model
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Hyperparameters
nz = 100
num_epochs = 10
lr = 0.0002
batch_size = 64

# Initialize models
generator = Generator(nz).to(device)
discriminator = Discriminator().to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for i, (data, _) in enumerate(dataloader):
        # Train Discriminator
        real_data = data.to(device)
        b_size = real_data.size(0)
        labels_real = torch.ones(b_size, 1, device=device)
        labels_fake = torch.zeros(b_size, 1, device=device)
        
        optimizer_d.zero_grad()
        output_real = discriminator(real_data).view(-1, 1)
        loss_real = criterion(output_real, labels_real)
        
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_data = generator(noise)
        output_fake = discriminator(fake_data.detach()).view(-1, 1)
        loss_fake = criterion(output_fake, labels_fake)
        
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        output_fake = discriminator(fake_data).view(-1, 1)
        loss_g = criterion(output_fake, labels_real)
        loss_g.backward()
        optimizer_g.step()

    # Print loss
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {loss_d:.4f}, Loss G: {loss_g:.4f}")

    # Save generated images for this epoch
    with torch.no_grad():
        noise = torch.randn(16, nz, 1, 1, device=device)
        generated_images = generator(noise).cpu()
        save_image(generated_images, os.path.join(output_dir, f"epoch_{epoch+1}.png"), nrow=4, normalize=True)

print("Training complete. Generated images are saved in the 'generated_images' folder.")
