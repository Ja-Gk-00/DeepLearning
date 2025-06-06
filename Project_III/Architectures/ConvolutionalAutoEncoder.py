import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from DataObjects import DataLoader
from ArchitectureModel import GeneratorBase


class ConvAutoEncoder(GeneratorBase, nn.Module):
    def __init__(self,
                 device: torch.device,
                 input_channels: int = 3,
                 latent_dim: int = 128,
                 base_channels: int = 64,
                 image_size: int = 64,
                 learning_rate: float = 1e-3):
        """
        Convolutional Autoencoder implementation

        Args:
            device: torch device for computation
            input_channels: number of input channels (3 for RGB, 1 for grayscale)
            latent_dim: dimension of the latent space
            base_channels: base number of channels for conv layers
            image_size: size of input images (assumed square)
            learning_rate: learning rate for optimizer
        """
        GeneratorBase.__init__(self, device)
        nn.Module.__init__(self)

        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.image_size = image_size
        self.learning_rate = learning_rate

        # Calculate the size after convolutions for the linear layer
        self.conv_output_size = self._calculate_conv_output_size()

        self.build_model()
        self._move_to_device()

    def _calculate_conv_output_size(self) -> int:
        """Calculate the flattened size after encoder convolutions"""
        # After 4 conv layers with stride 2, size is reduced by 2^4 = 16
        reduced_size = self.image_size // 16
        return self.base_channels * 8 * reduced_size * reduced_size

    def build_model(self) -> None:
        """Initialize encoder and decoder networks"""
        # Encoder
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

        # Latent space projection
        self.encoder_fc = nn.Linear(self.conv_output_size, self.latent_dim)
        self.decoder_fc = nn.Linear(self.latent_dim, self.conv_output_size)

        # Decoder
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)  # Flatten
        x = self.encoder_fc(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image"""
        batch_size = z.size(0)
        x = self.decoder_fc(z)
        # Reshape back to feature map
        reduced_size = self.image_size // 16
        x = x.view(batch_size, self.base_channels * 8, reduced_size, reduced_size)
        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder"""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed

    def sample_latent(self, num_samples: int) -> torch.Tensor:
        """Sample random latent vectors from standard normal distribution"""
        return torch.randn(num_samples, self.latent_dim, device=self.device)

    def generate(self, num_samples: int, **kwargs: Any) -> torch.Tensor:
        """Generate images from random latent vectors"""
        self.eval()
        with torch.no_grad():
            latent = self.sample_latent(num_samples)
            generated = self.decode(latent)
        return generated

    def train_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform one training step"""
        self.train()

        # Forward pass
        reconstructed = self.forward(batch)

        # Compute reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, batch, reduction='mean')

        # Backward pass
        self.optimizer.zero_grad()
        recon_loss.backward()
        self.optimizer.step()

        return {
            'reconstruction_loss': recon_loss.detach(),
            'total_loss': recon_loss.detach()
        }

    def train_architecture(self, data: DataLoader) -> None:
        """Train the autoencoder on the provided data"""
        self.train()

        for epoch in range(self.num_epochs):
            epoch_losses = []

            for batch_idx, batch in enumerate(data):
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # Assume first element is the data
                batch = batch.to(self.device)

                # Training step
                losses = self.train_step(batch)
                epoch_losses.append(losses['total_loss'].item())

                # Log progress
                if batch_idx % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{self.num_epochs}], '
                          f'Batch [{batch_idx}/{len(data)}], '
                          f'Loss: {losses["total_loss"].item():.4f}')

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f'Epoch [{epoch + 1}/{self.num_epochs}] completed, '
                  f'Average Loss: {avg_loss:.4f}')

            # Step scheduler if available
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.step()

    def evaluate(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Evaluate the model on a batch"""
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(batch)

            # Reconstruction loss
            recon_loss = F.mse_loss(reconstructed, batch, reduction='mean')

            # Mean Absolute Error
            mae = F.l1_loss(reconstructed, batch, reduction='mean')

            # Peak Signal-to-Noise Ratio (approximate)
            mse = recon_loss
            psnr = 20 * torch.log10(2.0 / torch.sqrt(mse + 1e-8))  # Assuming inputs in [-1,1]

        return {
            'reconstruction_loss': recon_loss,
            'mae': mae,
            'psnr': psnr
        }

    def configure_optimizers(self) -> Any:
        """Configure optimizer and scheduler"""
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.num_epochs = 100  # Default number of epochs

        return {
            'optimizer': self.optimizer,
            'scheduler': self.scheduler
        }

    def save_model(self, path: str) -> None:
        """Save model state dictionary"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'config': {
                'input_channels': self.input_channels,
                'latent_dim': self.latent_dim,
                'base_channels': self.base_channels,
                'image_size': self.image_size,
                'learning_rate': self.learning_rate
            }
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str, map_location: Optional[str] = None) -> None:
        """Load model state dictionary"""
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])

        if hasattr(self, 'optimizer') and checkpoint['optimizer_state_dict'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if hasattr(self, 'scheduler') and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Model loaded from {path}")

    def _move_to_device(self) -> None:
        """Move all modules to the specified device"""
        self.to(self.device)


# Example usage:
if __name__ == "__main__":
    # Initialize the autoencoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = ConvAutoEncoder(
        device=device,
        input_channels=3,  # RGB images
        latent_dim=128,
        base_channels=64,
        image_size=64,
        learning_rate=1e-3
    )

    # Configure optimizers
    autoencoder.configure_optimizers()

    # Example forward pass
    batch_size = 8
    dummy_input = torch.randn(batch_size, 3, 64, 64).to(device)

    # Reconstruction
    reconstructed = autoencoder(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")

    # Generation
    generated = autoencoder.generate(num_samples=4)
    print(f"Generated shape: {generated.shape}")

    # Evaluation
    eval_metrics = autoencoder.evaluate(dummy_input)
    print("Evaluation metrics:", eval_metrics)