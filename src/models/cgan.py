import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.gan import Generator, Discriminator
from utils.conformal import compute_icp_conformity_scores, compute_mondrian_conformity_scores, compute_cross_conformal_scores, compute_venn_abers_scores, compute_conformal_intervals
from utils.metrics import evaluate_metrics

class ConformalGAN:
    def __init__(self, input_dim, noise_dim, num_classes, device,
                 lambda_reg=0.1, mu_conform=0.1, 
                 weight_icp=0.4, weight_mondrian=0.3, 
                 weight_cross=0.2, weight_venn=0.1):
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.device = device
        self.lambda_reg = lambda_reg
        self.mu_conform = mu_conform
        self.weight_icp = weight_icp
        self.weight_mondrian = weight_mondrian
        self.weight_cross = weight_cross
        self.weight_venn = weight_venn
        self.G = Generator(noise_dim, num_classes, input_dim).to(device)
        self.D = Discriminator(input_dim, num_classes).to(device)
        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=0.0001)
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=0.0001)

    def train(self, real_data, real_labels, epochs=10000, batch_size=32):
        real_data = torch.tensor(real_data, device=self.device)
        real_labels = torch.tensor(real_labels, device=self.device)
        num_samples = real_data.size(0)

        for epoch in tqdm(range(epochs), desc="Training Conformal GAN (cGAN)"):
            # --- Train Discriminator ---
            self.D.zero_grad()
            idx = torch.randint(0, num_samples, (batch_size,))
            real_batch = real_data[idx]
            real_batch_labels = real_labels[idx]
            valid = torch.ones(batch_size, 1, device=self.device, requires_grad=False)
            fake = torch.zeros(batch_size, 1, device=self.device, requires_grad=False)

            # Real loss
            real_pred = self.D(real_batch, real_batch_labels)
            real_loss = self.criterion(real_pred, valid)

            # Fake loss
            z = torch.randn(batch_size, self.noise_dim, device=self.device)
            fake_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
            fake_batch = self.G(z, fake_labels)
            fake_pred = self.D(fake_batch.detach(), fake_labels)
            fake_loss = self.criterion(fake_pred, fake)

            # Conformity scores and regularization
            conf_real_icp = compute_icp_conformity_scores(real_batch.detach().cpu().numpy())
            conf_fake_icp = compute_icp_conformity_scores(fake_batch.detach().cpu().numpy())

            conf_real_mondrian = compute_mondrian_conformity_scores(real_batch.detach().cpu().numpy(), real_batch_labels.cpu().numpy())
            conf_fake_mondrian = compute_mondrian_conformity_scores(fake_batch.detach().cpu().numpy(), fake_labels.cpu().numpy())

            conf_real_cross = compute_cross_conformal_scores(real_batch.detach().cpu().numpy(), real_batch_labels.cpu().numpy())
            conf_fake_cross = compute_cross_conformal_scores(fake_batch.detach().cpu().numpy(), fake_labels.cpu().numpy())

            conf_real_venn = compute_venn_abers_scores(real_batch.detach().cpu().numpy()[:, 0], real_batch_labels.cpu().numpy())
            conf_fake_venn = compute_venn_abers_scores(fake_batch.detach().cpu().numpy()[:, 0], fake_labels.cpu().numpy())

            # Regularization losses
            reg_loss_icp = torch.mean(torch.abs(torch.tensor(conf_real_icp, dtype=torch.float32, device=self.device) - 
                                                torch.tensor(conf_fake_icp, dtype=torch.float32, device=self.device)))
            reg_loss_mondrian = torch.mean(torch.abs(torch.tensor(conf_real_mondrian, dtype=torch.float32, device=self.device) - 
                                                    torch.tensor(conf_fake_mondrian, dtype=torch.float32, device=self.device)))
            reg_loss_cross = torch.mean(torch.abs(torch.tensor(conf_real_cross, dtype=torch.float32, device=self.device) - 
                                                torch.tensor(conf_fake_cross, dtype=torch.float32, device=self.device)))
            reg_loss_venn = torch.mean(torch.abs(torch.tensor(conf_real_venn, dtype=torch.float32, device=self.device) - 
                                                torch.tensor(conf_fake_venn, dtype=torch.float32, device=self.device)))

            # Total regularization loss with weighted terms
            total_reg_loss = self.weight_icp * reg_loss_icp + self.weight_mondrian * reg_loss_mondrian + \
                            self.weight_cross * reg_loss_cross + self.weight_venn * reg_loss_venn

            # Ensure that the total loss is a scalar tensor that requires grad
            d_loss = real_loss + fake_loss - self.lambda_reg * total_reg_loss
            d_loss.backward()  # Perform the backward pass here

            # Update Discriminator's weights
            self.optimizer_D.step()

            # --- Train Generator ---
            self.G.zero_grad()
            z = torch.randn(batch_size, self.noise_dim, device=self.device)
            fake_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
            fake_batch = self.G(z, fake_labels)
            g_pred = self.D(fake_batch, fake_labels)
            g_loss = self.criterion(g_pred, valid)

            # Update this line to correctly compute the target_conform_icp:
            with torch.no_grad():
                target_conform_icp = torch.mean(torch.tensor(compute_icp_conformity_scores(real_batch.cpu().numpy()), dtype=torch.float32, device=self.device))

            # Ensure fake_batch is detached before converting to numpy
            conform_loss_icp = torch.mean((torch.tensor(compute_icp_conformity_scores(fake_batch.detach().cpu().numpy()), dtype=torch.float32, device=self.device) - target_conform_icp) ** 2)

            g_loss_total = g_loss + self.mu_conform * conform_loss_icp
            g_loss_total.backward()
            self.optimizer_G.step()

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

        return synthetic_data, synthetic_labels  # Return the synthetic data and labels
