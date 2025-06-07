import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from DataObjects import DataLoader
from Architectures.ArchitectureModel import GeneratorBase
from torchvision.utils import save_image
import os


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

        self.device = device  # Store device first
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.image_size = image_size
        self.learning_rate = learning_rate

        # Calculate the size after convolutions for the linear layer
        self.conv_output_size = self._calculate_conv_output_size()

        self.build_model()
        self._move_to_device()

        # Debug: Print device status after initialization
        print(f"Model initialized on device: {self.device}")
        print(f"First conv layer device: {next(self.encoder.parameters()).device}")

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

    def _ensure_device_compatibility(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input tensor is on the same device as the model"""
        model_device = next(self.parameters()).device
        if x.device != model_device:
            x = x.to(model_device)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        # Ensure input is on correct device
        x = self._ensure_device_compatibility(x)

        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)  # Flatten
        x = self.encoder_fc(x)
        return x

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder"""
        # Ensure input is on correct device
        x = self._ensure_device_compatibility(x)

        # Resize input if necessary (like DiffusionModel)
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = torch.nn.functional.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False
            )

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

        # Ensure batch is on correct device
        batch = self._ensure_device_compatibility(batch)

        # Resize input if necessary (like DiffusionModel)
        if batch.shape[-2:] != (self.image_size, self.image_size):
            batch = torch.nn.functional.interpolate(
                batch,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False
            )

        # Forward pass
        reconstructed = self.forward(batch)

        # Compute reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, batch, reduction='mean')

        return {
            'loss': recon_loss,  # Changed key to match DiffusionModel
            'reconstruction_loss': recon_loss,
            'total_loss': recon_loss
        }

    def train_architecture(self, dataloader: DataLoader, epochs: int) -> None:
        """Train the autoencoder on the provided data - Compatible with DiffusionModel style"""
        optimizer = self.configure_optimizers()

        for epoch in range(epochs):
            epoch_losses = []

            for batch_idx, batch_obj in enumerate(dataloader):
                # Extract images from the DataLoader format: [(img, _), (img, _), ...]
                # Same as DiffusionModel approach
                imgs = torch.stack([img for img, _ in batch_obj])
                # Ensure images are moved to correct device
                imgs = self._ensure_device_compatibility(imgs)

                # Training step
                losses = self.train_step(imgs)

                # Backward pass
                optimizer.zero_grad()
                losses['loss'].backward()
                optimizer.step()

                epoch_losses.append(losses['loss'].item())

                # Log progress
                if batch_idx % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], '
                          f'Batch [{batch_idx}/{len(dataloader)}], '
                          f'Loss: {losses["loss"].item():.4f}')

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f'Epoch [{epoch + 1}/{epochs}] completed, '
                  f'Average Loss: {avg_loss:.4f}')

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

            reconstructed = self.forward(batch)

            # Reconstruction loss
            recon_loss = F.mse_loss(reconstructed, batch, reduction='mean')

            # Mean Absolute Error
            mae = F.l1_loss(reconstructed, batch, reduction='mean')

            # Peak Signal-to-Noise Ratio (approximate)
            mse = recon_loss
            psnr = 20 * torch.log10(2.0 / torch.sqrt(mse + 1e-8))  # Assuming inputs in [-1,1]

        return {
            'loss': recon_loss,  # Added to match DiffusionModel
            'reconstruction_loss': recon_loss,
            'mae': mae,
            'psnr': psnr
        }

    def configure_optimizers(self) -> Any:
        """Configure optimizer - simplified like DiffusionModel"""
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def save_model(self, path: str) -> None:
        """Save model state dictionary - simplified like DiffusionModel"""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str, map_location: Optional[str] = None) -> None:
        """Load model state dictionary - simplified like DiffusionModel"""
        # If no map_location specified and we have a device, use it
        if map_location is None:
            map_location = self.device

        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)

        # Ensure model is on correct device after loading
        self._move_to_device()
        print(f"Model loaded from {path} and moved to {self.device}")

    def _move_to_device(self) -> None:
        """Move all modules to the specified device"""
        self.to(self.device)

    def to(self, device: torch.device) -> 'ConvAutoEncoder':
        """Override to method to update internal device reference"""
        self.device = device
        super().to(device)
        return self

    def cuda(self, device: Optional[int] = None) -> 'ConvAutoEncoder':
        """Override cuda method to update internal device reference"""
        if device is None:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device(f'cuda:{device}')
        super().cuda(device)
        return self

    def cpu(self) -> 'ConvAutoEncoder':
        """Override cpu method to update internal device reference"""
        self.device = torch.device('cpu')
        super().cpu()
        return self

    def save_generated_images(self, images: torch.Tensor, folder_path: str, prefix: str = "generated_image",
                              normalize: bool = True) -> None:
        """
        Saves a batch of generated images as individual PNG files.

        Args:
            images: A batch of images (tensor of shape [N, C, H, W]) to save.
                    The pixel values are expected to be in the range [-1, 1] due to Tanh activation.
            folder_path: The directory where images will be saved. Will be created if it doesn't exist.
            prefix: A prefix for the filenames of the saved images (e.g., "generated_image_0.png").
            normalize: If True, pixel values are normalized to [0, 1] before saving. Set to False if your
                       output is already in [0, 1] or [0, 255].
        """
        os.makedirs(folder_path, exist_ok=True)
        for i, img in enumerate(images):
            # Denormalize from [-1, 1] to [0, 1] for saving if normalize is True
            if normalize:
                img = (img + 1) / 2
            filepath = os.path.join(folder_path, f"{prefix}_{i:04d}.png")
            save_image(img.cpu(), filepath)
        print(f"Saved {len(images)} generated images to {folder_path}")

    @classmethod
    def from_pretrained(cls, device: torch.device, **kwargs) -> "ConvAutoEncoder":
        """Create and initialize model - following DiffusionModel pattern"""
        model = cls(device, **kwargs)
        # Ensure model is definitely on the correct device
        model.to(device)
        print(f"Model moved to device: {device}")
        print(f"Model parameters on device: {next(model.parameters()).device}")
        return model


# Example usage:
if __name__ == "__main__":
    # Initialize the autoencoder - using from_pretrained like DiffusionModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = ConvAutoEncoder.from_pretrained(
        device=device,
        input_channels=3,  # RGB images
        latent_dim=128,
        base_channels=64,
        image_size=64,
        learning_rate=1e-3
    )

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

    # --- New functionality: Saving generated images ---
    output_dir = "generated_images_cae"
    autoencoder.save_generated_images(generated, output_dir, prefix="my_generated_image")

    # You can also save reconstructed images
    reconstructed_output_dir = "reconstructed_images_cae"
    autoencoder.save_generated_images(reconstructed, reconstructed_output_dir, prefix="reconstructed_image")