import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, output_dim):
        super(Generator, self).__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, z, labels):
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        x = torch.cat([z, labels_onehot], dim=1)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim + num_classes, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        x = torch.cat([x, labels_onehot], dim=1)
        return self.net(x)

class StandardGAN:
    def __init__(self, input_dim, noise_dim, num_classes, device):
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.device = device
        self.G = Generator(noise_dim, num_classes, input_dim).to(device)
        self.D = Discriminator(input_dim, num_classes).to(device)
        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=0.0001)
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=0.0001)

    def train(self, real_data, real_labels, epochs=10000, batch_size=32):
        real_data = torch.tensor(real_data, device=self.device)
        real_labels = torch.tensor(real_labels, device=self.device)
        num_samples = real_data.size(0)
        
        for epoch in tqdm(range(epochs), desc="Training Standard GAN"):
            # --- Train Discriminator ---
            self.D.zero_grad()
            idx = torch.randint(0, num_samples, (batch_size,))
            real_batch = real_data[idx]
            real_batch_labels = real_labels[idx]
            valid = torch.ones(batch_size, 1, device=self.device)
            fake = torch.zeros(batch_size, 1, device=self.device)
            
            # Real loss
            real_pred = self.D(real_batch, real_batch_labels)
            real_loss = self.criterion(real_pred, valid)
            
            # Fake loss
            z = torch.randn(batch_size, self.noise_dim, device=self.device)
            fake_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
            fake_batch = self.G(z, fake_labels)
            fake_pred = self.D(fake_batch.detach(), fake_labels)
            fake_loss = self.criterion(fake_pred, fake)
            
            d_loss = real_loss + fake_loss
            d_loss.backward()
            self.optimizer_D.step()
            
            # --- Train Generator ---
            self.G.zero_grad()
            z = torch.randn(batch_size, self.noise_dim, device=self.device)
            fake_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
            fake_batch = self.G(z, fake_labels)
            g_pred = self.D(fake_batch, fake_labels)
            g_loss = self.criterion(g_pred, valid)
            g_loss.backward()
            self.optimizer_G.step()
            
            if epoch % 200 == 0:
                tqdm.write(f"Epoch {epoch}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")
        
        # Generate synthetic dataset (size equal to real dataset)
        synthetic_data = []
        synthetic_labels = []
        with torch.no_grad():
            for _ in range(num_samples // batch_size + 1):
                z = torch.randn(batch_size, self.noise_dim, device=self.device)
                fake_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
                gen_data = self.G(z, fake_labels)
                synthetic_data.append(gen_data.cpu())
                synthetic_labels.append(fake_labels.cpu())
        synthetic_data = torch.cat(synthetic_data, dim=0)[:num_samples].numpy()
        synthetic_labels = torch.cat(synthetic_labels, dim=0)[:num_samples].numpy()
        return synthetic_data, synthetic_labels
