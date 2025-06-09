import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from DataObjects import DataLoader
from Architectures.ArchitectureModel import GeneratorBase
from torchvision.utils import save_image
import os


class ConvVAE(GeneratorBase, nn.Module):
    def __init__(self,
                 device: torch.device,
                 input_channels: int = 3,
                 latent_dim: int = 96,
                 base_channels: int = 64,
                 image_size: int = 64,
                 learning_rate: float = 2e-4,
                 beta: float = 1.0):
        """
        Convolutional Variational Autoencoder implementation for cat image generation

        Args:
            device: torch device for computation
            input_channels: number of input channels (3 for RGB, 1 for grayscale)
            latent_dim: dimension of the latent space
            base_channels: base number of channels for conv layers
            image_size: size of input images (assumed square)
            learning_rate: learning rate for optimizer
            beta: weight for KL divergence loss (beta-VAE)
        """
        GeneratorBase.__init__(self, device)
        nn.Module.__init__(self)

        self.device = device
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.beta = beta  # Beta parameter for beta-VAE

        # Calculate the size after convolutions for the linear layer
        self.conv_output_size = self._calculate_conv_output_size()

        self.build_model()
        self._move_to_device()

        # Debug: Print device status after initialization
        print(f"VAE initialized on device: {self.device}")
        print(f"First conv layer device: {next(self.encoder.parameters()).device}")

    def _calculate_conv_output_size(self) -> int:
        """Calculate the flattened size after encoder convolutions"""
        # After 4 conv layers with stride 2, size is reduced by 2^4 = 16
        reduced_size = self.image_size // 16
        return self.base_channels * 8 * reduced_size * reduced_size

    def build_model(self) -> None:
        """Initialize encoder and decoder networks with VAE structure"""
        # Encoder (same as before)
        self.encoder = nn.Sequential(
            # Layer 1: input_channels -> base_channels
            nn.Conv2d(self.input_channels, self.base_channels, 4, 2, 1),
            nn.BatchNorm2d(self.base_channels),
            nn.ReLU(inplace=True),

            # Layer 2: base_channels -> base_channels*2
            nn.Conv2d(self.base_channels, self.base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(self.base_channels * 2),
            nn.ReLU(inplace=True),

            # Layer 3: base_channels*2 -> base_channels*4
            nn.Conv2d(self.base_channels * 2, self.base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(self.base_channels * 4),
            nn.ReLU(inplace=True),

            # Layer 4: base_channels*4 -> base_channels*8
            nn.Conv2d(self.base_channels * 4, self.base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(self.base_channels * 8),
            nn.ReLU(inplace=True),
        )

        # VAE latent space projection - separate mu and logvar
        self.fc_mu = nn.Linear(self.conv_output_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_size, self.latent_dim)
        self.decoder_fc = nn.Linear(self.latent_dim, self.conv_output_size)

        # Decoder (same as before)
        self.decoder = nn.Sequential(
            # Layer 1: base_channels*8 -> base_channels*4
            nn.ConvTranspose2d(self.base_channels * 8, self.base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(self.base_channels * 4),
            nn.ReLU(inplace=True),

            # Layer 2: base_channels*4 -> base_channels*2
            nn.ConvTranspose2d(self.base_channels * 4, self.base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(self.base_channels * 2),
            nn.ReLU(inplace=True),

            # Layer 3: base_channels*2 -> base_channels
            nn.ConvTranspose2d(self.base_channels * 2, self.base_channels, 4, 2, 1),
            nn.BatchNorm2d(self.base_channels),
            nn.ReLU(inplace=True),

            # Layer 4: base_channels -> input_channels
            nn.ConvTranspose2d(self.base_channels, self.input_channels, 4, 2, 1),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def _ensure_device_compatibility(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input tensor is on the same device as the model"""
        model_device = next(self.parameters()).device
        if x.device != model_device:
            x = x.to(model_device)
        return x

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space parameters (mu, logvar)"""
        # Ensure input is on correct device
        x = self._ensure_device_compatibility(x)

        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)  # Flatten

        # Get mean and log variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * epsilon"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, use the mean
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image"""
        # Ensure input is on correct device
        z = self._ensure_device_compatibility(z)

        batch_size = z.size(0)
        x = self.decoder_fc(z)
        # Reshape back to feature map
        reduced_size = self.image_size // 16
        x = x.view(batch_size, self.base_channels * 8, reduced_size, reduced_size)
        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE"""
        # Ensure input is on correct device
        x = self._ensure_device_compatibility(x)

        # Resize input if necessary
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = torch.nn.functional.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False
            )

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)

        return reconstructed, mu, logvar

    def sample_latent(self, num_samples: int, temperature: float = 1.0) -> torch.Tensor:
        """Sample latent vectors from standard normal distribution"""
        return torch.randn(num_samples, self.latent_dim, device=self.device) * temperature

    def generate(self, num_samples: int, temperature: float = 1.0, **kwargs: Any) -> torch.Tensor:
        """Generate images from random latent vectors"""
        self.eval()
        with torch.no_grad():
            latent = self.sample_latent(num_samples, temperature)
            generated = self.decode(latent)
        return generated

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """Interpolate between two images in latent space"""
        self.eval()
        with torch.no_grad():
            # Encode both images
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)

            # Create interpolation steps
            alphas = torch.linspace(0, 1, num_steps, device=self.device).view(-1, 1)
            interpolated_z = mu1 * (1 - alphas) + mu2 * alphas

            # Decode interpolated latent vectors
            interpolated_images = self.decode(interpolated_z)

        return interpolated_images

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between latent distribution and standard normal"""
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()

    def train_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform one training step with VAE loss"""
        self.train()

        # Ensure batch is on correct device
        batch = self._ensure_device_compatibility(batch)

        # Resize input if necessary
        if batch.shape[-2:] != (self.image_size, self.image_size):
            batch = torch.nn.functional.interpolate(
                batch,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False
            )

        # Forward pass
        reconstructed, mu, logvar = self.forward(batch)

        # Compute losses
        # Reconstruction loss (MSE or BCE)
        recon_loss = F.mse_loss(reconstructed, batch, reduction='mean')

        # KL divergence loss
        kl_loss = self.kl_divergence(mu, logvar)

        # Total VAE loss
        total_loss = recon_loss + self.beta * kl_loss

        return {
            'loss': total_loss,  # Main loss for backward pass
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss
        }

    def train_architecture(self, dataloader: DataLoader, epochs: int) -> None:
        """Train the VAE on the provided data"""
        optimizer = self.configure_optimizers()

        for epoch in range(epochs):
            epoch_losses = {'total': [], 'recon': [], 'kl': []}

            for batch_idx, batch_obj in enumerate(dataloader):
                # Extract images from the DataLoader format
                imgs = torch.stack([img for img, _ in batch_obj])
                imgs = self._ensure_device_compatibility(imgs)

                # Training step
                losses = self.train_step(imgs)

                # Backward pass
                optimizer.zero_grad()
                losses['loss'].backward()
                optimizer.step()

                # Store losses
                epoch_losses['total'].append(losses['total_loss'].item())
                epoch_losses['recon'].append(losses['reconstruction_loss'].item())
                epoch_losses['kl'].append(losses['kl_loss'].item())

                # Log progress
                if batch_idx % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], '
                          f'Batch [{batch_idx}/{len(dataloader)}], '
                          f'Total Loss: {losses["total_loss"].item():.4f}, '
                          f'Recon Loss: {losses["reconstruction_loss"].item():.4f}, '
                          f'KL Loss: {losses["kl_loss"].item():.4f}')

            # Epoch summary
            avg_total = sum(epoch_losses['total']) / len(epoch_losses['total'])
            avg_recon = sum(epoch_losses['recon']) / len(epoch_losses['recon'])
            avg_kl = sum(epoch_losses['kl']) / len(epoch_losses['kl'])

            print(f'Epoch [{epoch + 1}/{epochs}] completed, '
                  f'Avg Total Loss: {avg_total:.4f}, '
                  f'Avg Recon Loss: {avg_recon:.4f}, '
                  f'Avg KL Loss: {avg_kl:.4f}')

    def evaluate(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Evaluate the model on a batch"""
        self.eval()
        with torch.no_grad():
            # Ensure batch is on correct device
            batch = self._ensure_device_compatibility(batch)

            # Resize input if necessary
            if batch.shape[-2:] != (self.image_size, self.image_size):
                batch = torch.nn.functional.interpolate(
                    batch,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False
                )

            reconstructed, mu, logvar = self.forward(batch)

            # Compute losses
            recon_loss = F.mse_loss(reconstructed, batch, reduction='mean')
            kl_loss = self.kl_divergence(mu, logvar)
            total_loss = recon_loss + self.beta * kl_loss

            # Additional metrics
            mae = F.l1_loss(reconstructed, batch, reduction='mean')
            mse = recon_loss
            psnr = 20 * torch.log10(2.0 / torch.sqrt(mse + 1e-8))

        return {
            'loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss,
            'mae': mae,
            'psnr': psnr
        }

    def configure_optimizers(self) -> Any:
        """Configure optimizer"""
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def save_model(self, path: str) -> None:
        """Save model state dictionary"""
        torch.save({
            'state_dict': self.state_dict(),
            'latent_dim': self.latent_dim,
            'base_channels': self.base_channels,
            'image_size': self.image_size,
            'input_channels': self.input_channels,
            'beta': self.beta
        }, path)
        print(f"VAE model saved to {path}")

    def load_model(self, path: str, map_location: Optional[str] = None) -> None:
        """Load model state dictionary"""
        if map_location is None:
            map_location = self.device

        checkpoint = torch.load(path, map_location=map_location)

        # Load hyperparameters if available
        if 'latent_dim' in checkpoint:
            self.latent_dim = checkpoint['latent_dim']
            self.base_channels = checkpoint['base_channels']
            self.image_size = checkpoint['image_size']
            self.input_channels = checkpoint['input_channels']
            self.beta = checkpoint.get('beta', 1.0)

            # Rebuild model with loaded parameters
            self.build_model()

        # Load state dict
        state_dict = checkpoint.get('state_dict', checkpoint)
        self.load_state_dict(state_dict)

        self._move_to_device()
        print(f"VAE model loaded from {path} and moved to {self.device}")

    def _move_to_device(self) -> None:
        """Move all modules to the specified device"""
        self.to(self.device)

    def to(self, device: torch.device) -> 'ConvVAE':
        """Override to method to update internal device reference"""
        self.device = device
        super().to(device)
        return self

    def cuda(self, device: Optional[int] = None) -> 'ConvVAE':
        """Override cuda method to update internal device reference"""
        if device is None:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device(f'cuda:{device}')
        super().cuda(device)
        return self

    def cpu(self) -> 'ConvVAE':
        """Override cpu method to update internal device reference"""
        self.device = torch.device('cpu')
        super().cpu()
        return self

    def save_generated_images(self, images: torch.Tensor, folder_path: str, prefix: str = "generated_image",
                              normalize: bool = True) -> None:
        """Save a batch of generated images as individual PNG files"""
        os.makedirs(folder_path, exist_ok=True)
        for i, img in enumerate(images):
            if normalize:
                img = (img + 1) / 2
            filepath = os.path.join(folder_path, f"{prefix}_{i:04d}.png")
            save_image(img.cpu(), filepath)
        print(f"Saved {len(images)} generated images to {folder_path}")

    @classmethod
    def from_pretrained(cls, device: torch.device, **kwargs) -> "ConvVAE":
        """Create and initialize VAE model"""
        model = cls(device, **kwargs)
        model.to(device)
        print(f"VAE model moved to device: {device}")
        print(f"VAE model parameters on device: {next(model.parameters()).device}")
        return model
