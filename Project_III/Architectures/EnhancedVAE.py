import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import os
from torchvision.utils import save_image
from torchvision import datasets, transforms  # Required for example usage
from torch.utils.data import DataLoader  # Required for example usage


class EnhancedConvVAE(nn.Module):
    def __init__(self,
                 device: torch.device,
                 input_channels: int = 3,
                 latent_dim: int = 256,  # Increased for more diversity
                 base_channels: int = 64,
                 image_size: int = 64,
                 learning_rate: float = 1e-4,  # Reduced for stability
                 beta: float = 0.5,  # Reduced to prevent posterior collapse (for non-capacity mode)
                 gamma: float = 1000,  # For spectral normalization
                 use_spectral_norm: bool = True):
        """
        Enhanced VAE with improved diversity mechanisms and integrated training loop.
        """
        super().__init__()
        self.device = device
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.gamma = gamma
        self.use_spectral_norm = use_spectral_norm

        # Add capacity scheduling for beta-VAE
        self.capacity = 0.0
        self.max_capacity = 25.0
        self.capacity_change_duration = 100000
        self.global_step = 0  # Moved here from train_step, will be updated in train_architecture

        self.conv_output_size = self._calculate_conv_output_size()
        self.build_model()
        self._move_to_device()  # Ensure model is on device after build

        print(f"EnhancedConvVAE initialized on device: {self.device}")

    def _calculate_conv_output_size(self) -> int:
        reduced_size = self.image_size // 16
        return self.base_channels * 8 * reduced_size * reduced_size

    def _add_spectral_norm(self, layer):
        """Add spectral normalization if enabled"""
        if self.use_spectral_norm and isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            return nn.utils.spectral_norm(layer)
        return layer

    def build_model(self) -> None:
        """Build encoder and decoder with enhanced architecture"""
        self.encoder = nn.ModuleList([
            nn.Sequential(
                self._add_spectral_norm(nn.Conv2d(self.input_channels, self.base_channels, 4, 2, 1)),
                nn.BatchNorm2d(self.base_channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.1)
            ),
            nn.Sequential(
                self._add_spectral_norm(nn.Conv2d(self.base_channels, self.base_channels * 2, 4, 2, 1)),
                nn.BatchNorm2d(self.base_channels * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.1)
            ),
            nn.Sequential(
                self._add_spectral_norm(nn.Conv2d(self.base_channels * 2, self.base_channels * 4, 4, 2, 1)),
                nn.BatchNorm2d(self.base_channels * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.1)
            ),
            nn.Sequential(
                self._add_spectral_norm(nn.Conv2d(self.base_channels * 4, self.base_channels * 8, 4, 2, 1)),
                nn.BatchNorm2d(self.base_channels * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.1)
            )
        ])

        self.fc_mu = self._add_spectral_norm(nn.Linear(self.conv_output_size, self.latent_dim))
        self.fc_logvar = self._add_spectral_norm(nn.Linear(self.conv_output_size, self.latent_dim))

        nn.init.xavier_normal_(self.fc_mu.weight)
        nn.init.xavier_normal_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_mu.bias, 0)
        nn.init.constant_(self.fc_logvar.bias, -1)

        self.decoder_fc = self._add_spectral_norm(nn.Linear(self.latent_dim, self.conv_output_size))

        self.decoder = nn.Sequential(
            self._add_spectral_norm(nn.ConvTranspose2d(self.base_channels * 8, self.base_channels * 4, 4, 2, 1)),
            nn.BatchNorm2d(self.base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),

            self._add_spectral_norm(nn.ConvTranspose2d(self.base_channels * 4, self.base_channels * 2, 4, 2, 1)),
            nn.BatchNorm2d(self.base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),

            self._add_spectral_norm(nn.ConvTranspose2d(self.base_channels * 2, self.base_channels, 4, 2, 1)),
            nn.BatchNorm2d(self.base_channels),
            nn.ReLU(inplace=True),

            self._add_spectral_norm(nn.ConvTranspose2d(self.base_channels, self.input_channels, 4, 2, 1)),
            nn.Tanh()
        )

    def _ensure_device_compatibility(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input tensor is on the same device as the model"""
        model_device = next(self.parameters()).device
        if x.device != model_device:
            x = x.to(model_device)
        return x

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced encoding with feature extraction"""
        x = self._ensure_device_compatibility(x)
        batch_size = x.size(0)
        for layer in self.encoder:
            x = layer(x)
        x = x.view(batch_size, -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, -10, 10)  # Clamp logvar
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor,
                       temperature: float = 1.0) -> torch.Tensor:
        """Enhanced reparameterization with temperature control"""
        if self.training:
            std = torch.exp(0.5 * logvar) * temperature
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            if temperature > 0:
                std = torch.exp(0.5 * logvar) * temperature
                eps = torch.randn_like(std)
                return mu + eps * std
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Enhanced decoding"""
        z = self._ensure_device_compatibility(z)
        batch_size = z.size(0)
        x = self.decoder_fc(z)
        reduced_size = self.image_size // 16
        x = x.view(batch_size, self.base_channels * 8, reduced_size, reduced_size)
        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with temperature control"""
        x = self._ensure_device_compatibility(x)
        # Resize input if necessary, similar to ConvVAE
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = torch.nn.functional.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False
            )
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, temperature)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between latent distribution and standard normal"""
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()

    def compute_capacity_loss(self, kl_loss: torch.Tensor) -> torch.Tensor:
        """Compute capacity-constrained loss for better training"""
        # Capacity is updated globally in train_architecture, here just compute loss
        return self.gamma * torch.abs(kl_loss - self.capacity)

    def train_step(self, batch: torch.Tensor, use_capacity: bool = True) -> Dict[str, torch.Tensor]:
        """
        Perform one training step for the VAE.
        This method computes losses but does NOT perform backpropagation or optimizer steps.
        It's designed to be called by `train_architecture`.
        """
        self.train()  # Ensure model is in training mode

        # Forward pass with slight temperature for training diversity
        reconstructed, mu, logvar = self.forward(batch, temperature=1.05)

        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, batch, reduction='mean')

        # KL divergence
        kl_loss = self.kl_divergence(mu, logvar)

        # Total loss with capacity scheduling
        if use_capacity:
            capacity_loss = self.compute_capacity_loss(kl_loss)
            total_loss = recon_loss + capacity_loss
        else:
            total_loss = recon_loss + self.beta * kl_loss  # Use fixed beta if no capacity scheduling

        # Additional diversity regularization (applied probabilistically)
        # Note: This affects 'total_loss' and implicitly influences gradients
        if self.training and torch.rand(1).item() < 0.1:  # 10% of the time
            batch_size = mu.size(0)
            if batch_size > 1:
                mu_expanded = mu.unsqueeze(1).expand(-1, batch_size, -1)
                mu_t_expanded = mu.unsqueeze(0).expand(batch_size, -1, -1)
                distances = torch.norm(mu_expanded - mu_t_expanded + 1e-8, dim=2)
                non_self_mask = (distances > 0)
                if non_self_mask.sum() > 0:
                    diversity_weight_train = 0.01
                    repulsion = torch.where(non_self_mask, 1.0 / (distances), torch.tensor(0.0, device=self.device))
                    diversity_loss = diversity_weight_train * torch.mean(repulsion)
                    total_loss += diversity_loss

        return {
            'loss': total_loss,  # Main loss for backward pass in train_architecture
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss,
            'capacity': torch.tensor(self.capacity)  # Current capacity value
        }

    def train_architecture(self, dataloader: 'DataLoader', epochs: int, use_capacity: bool = True) -> Dict[
        str, List[float]]:
        optimizer = self.configure_optimizers()

        history = {'total_loss': [], 'recon_loss': [], 'kl_loss': [], 'capacity': []}

        print("Starting EnhancedConvVAE training...")
        # Keep track of global batch index for logging and capacity scheduling
        global_batch_idx = 0

        for epoch in range(epochs):
            self.train()
            epoch_losses = {'total': [], 'recon': [], 'kl': [], 'capacity': []}

            # Iterate through the custom DataLoader to get DataBatch objects
            for batch_object_idx, data_batch_object in enumerate(dataloader):
                # Now iterate through the DataBatch object to get individual (tensor, label) pairs
                # You'll collect all tensors for the current batch_object into a single tensor
                # before passing to train_step.
                batch_tensors = []
                # Your DataBatch yields (tensor, label), we only need the tensor for VAE input
                for tensor, _ in data_batch_object:
                    batch_tensors.append(tensor)

                # Stack the list of tensors into a single batch tensor
                # Ensure all tensors in the list have the same shape for stacking
                try:
                    current_batch_data = torch.stack(batch_tensors)
                except RuntimeError as e:
                    print(f"Error stacking tensors in batch {batch_object_idx}: {e}")
                    print("Skipping this batch.")
                    continue  # Skip this batch if stacking fails

                # Update global_step for capacity scheduling (per actual batch passed to model)
                self.global_step += 1
                global_batch_idx += 1  # This acts as the equivalent of `batch_idx` in your original loop

                # Update capacity value
                if self.global_step < self.capacity_change_duration:
                    self.capacity = min(self.max_capacity,
                                        self.max_capacity * self.global_step / self.capacity_change_duration)
                else:
                    self.capacity = self.max_capacity

                current_batch_data = self._ensure_device_compatibility(current_batch_data)

                if current_batch_data.shape[-2:] != (self.image_size, self.image_size):
                    current_batch_data = F.interpolate(
                        current_batch_data,
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False
                    )

                optimizer.zero_grad()

                loss_dict = self.train_step(current_batch_data, use_capacity=use_capacity)

                loss_dict['loss'].backward()
                optimizer.step()

                epoch_losses['total'].append(loss_dict['total_loss'].item())
                epoch_losses['recon'].append(loss_dict['reconstruction_loss'].item())
                epoch_losses['kl'].append(loss_dict['kl_loss'].item())
                epoch_losses['capacity'].append(loss_dict['capacity'].item())

                # Log progress using the global_batch_idx
                if global_batch_idx % 100 == 0:  # Use global_batch_idx here
                    print(f'Epoch [{epoch + 1}/{epochs}], '
                          f'Batch [{global_batch_idx}/{len(dataloader) * dataloader.batch_size // dataloader.batch_size}], '  # Total batches in terms of original batch size
                          f'Total Loss: {loss_dict["total_loss"].item():.4f}, '
                          f'Recon Loss: {loss_dict["reconstruction_loss"].item():.4f}, '
                          f'KL Loss: {loss_dict["kl_loss"].item():.4f}, '
                          f'Capacity: {self.capacity:.2f}')

            avg_total = np.mean(epoch_losses['total'])
            avg_recon = np.mean(epoch_losses['recon'])
            avg_kl = np.mean(epoch_losses['kl'])
            avg_capacity_epoch = np.mean(epoch_losses['capacity'])

            history['total_loss'].append(avg_total)
            history['recon_loss'].append(avg_recon)
            history['kl_loss'].append(avg_kl)
            history['capacity'].append(avg_capacity_epoch)

            print(f'Epoch [{epoch + 1}/{epochs}] completed, '
                  f'Avg Total Loss: {avg_total:.4f}, '
                  f'Avg Recon Loss: {avg_recon:.4f}, '
                  f'Avg KL Loss: {avg_kl:.4f}, '
                  f'Avg Capacity: {avg_capacity_epoch:.2f}')

        print("EnhancedConvVAE training complete.")
        return history

    def evaluate(self, dataloader: 'DataLoader') -> Dict[str, float]:
        self.eval()
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_loss = 0.0
        total_mae = 0.0
        total_psnr = 0.0
        num_processed_batches = 0  # Track actual number of batches processed

        with torch.no_grad():
            for data_batch_object in dataloader:  # Iterate to get DataBatch objects
                batch_tensors = []
                for tensor, _ in data_batch_object:  # Iterate DataBatch to get (tensor, label)
                    batch_tensors.append(tensor)

                try:
                    current_batch_data = torch.stack(batch_tensors)
                except RuntimeError as e:
                    print(f"Error stacking tensors in evaluation batch: {e}")
                    print("Skipping this evaluation batch.")
                    continue

                current_batch_data = self._ensure_device_compatibility(current_batch_data)

                if current_batch_data.shape[-2:] != (self.image_size, self.image_size):
                    current_batch_data = F.interpolate(
                        current_batch_data,
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False
                    )

                reconstructed, mu, logvar = self.forward(current_batch_data)

                recon_loss = F.mse_loss(reconstructed, current_batch_data, reduction='mean')
                kl_loss = self.kl_divergence(mu, logvar)

                current_beta_for_eval = self.beta if not hasattr(self,
                                                                 'capacity') or self.capacity == 0 else self.capacity / self.gamma
                total_loss_batch = recon_loss + current_beta_for_eval * kl_loss

                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_loss += total_loss_batch.item()

                mae = F.l1_loss(reconstructed, current_batch_data, reduction='mean')
                mse = recon_loss
                psnr = 20 * torch.log10(2.0 / torch.sqrt(mse + 1e-8))

                total_mae += mae.item()
                total_psnr += psnr.item()
                num_processed_batches += 1

        if num_processed_batches == 0:
            return {
                'reconstruction_loss': 0.0, 'kl_loss': 0.0, 'total_loss': 0.0,
                'mae': 0.0, 'psnr': 0.0
            }

        avg_recon_loss = total_recon_loss / num_processed_batches
        avg_kl_loss = total_kl_loss / num_processed_batches
        avg_total_loss = total_loss / num_processed_batches
        avg_mae = total_mae / num_processed_batches
        avg_psnr = total_psnr / num_processed_batches

        return {
            'reconstruction_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss,
            'total_loss': avg_total_loss,
            'mae': avg_mae,
            'psnr': avg_psnr
        }

    def configure_optimizers(self) -> Any:
        """Configure optimizer with AdamW for better settings"""
        return optim.AdamW(self.parameters(), lr=self.learning_rate,
                           weight_decay=1e-5, betas=(0.5, 0.999))

    def sample_latent(self, num_samples: int, temperature: float = 1.0,
                      method: str = 'normal') -> torch.Tensor:
        """Sample latent vectors from standard normal distribution or other methods"""
        if method == 'normal':
            return torch.randn(num_samples, self.latent_dim, device=self.device) * temperature
        elif method == 'uniform':
            return (torch.rand(num_samples, self.latent_dim, device=self.device) - 0.5) * 2 * temperature * 1.7
        elif method == 'mixture':
            k = 5
            component = torch.randint(0, k, (num_samples,), device=self.device)
            centers = torch.randn(k, self.latent_dim, device=self.device) * 2
            samples = torch.randn(num_samples, self.latent_dim, device=self.device)
            for i in range(num_samples):
                samples[i] += centers[component[i]]
            return samples * temperature
        elif method == 'hypersphere':
            samples = torch.randn(num_samples, self.latent_dim, device=self.device)
            norms = torch.norm(samples, dim=1, keepdim=True)
            radius = torch.rand(num_samples, 1, device=self.device) ** (1 / self.latent_dim)
            return samples / norms * radius * temperature * 3
        else:
            return torch.randn(num_samples, self.latent_dim, device=self.device) * temperature

    def generate(self, num_samples: int, temperature: float = 1.0, **kwargs: Any) -> torch.Tensor:
        """
        Generate images from random latent vectors using the default 'normal' sampling.
        This provides a consistent interface with other generator models.
        For diverse generation, use generate_diverse or generate_with_diversity_loss.
        """
        self.eval()
        with torch.no_grad():
            latent = self.sample_latent(num_samples, temperature, method='normal')
            generated = self.decode(latent)
        return generated

    def generate_diverse(self, num_samples: int, temperature: float = 1.2,
                         methods: list = None) -> torch.Tensor:
        """Generate diverse samples using multiple sampling methods"""
        if methods is None:
            methods = ['normal', 'uniform', 'mixture', 'hypersphere']

        self.eval()
        samples_per_method = num_samples // len(methods)
        remainder = num_samples % len(methods)
        all_samples = []

        with torch.no_grad():
            for i, method in enumerate(methods):
                n_samples = samples_per_method + (1 if i < remainder else 0)
                if n_samples > 0:
                    latent = self.sample_latent(n_samples, temperature, method)
                    generated = self.decode(latent)
                    all_samples.append(generated)
        return torch.cat(all_samples, dim=0) if all_samples else torch.empty(0)

    def generate_with_diversity_loss(self, num_samples: int, diversity_weight: float = 0.1):
        """Generate samples with explicit diversity encouragement"""
        self.eval()
        with torch.no_grad():  # Use no_grad here as this is for generation not training
            z = self.sample_latent(num_samples, temperature=1.2, method='mixture')

            # Temporarily enable gradients for diversity regularization if needed for this generation method
            # This is a bit unusual for inference, normally you'd just sample.
            # If the diversity_loss logic involves gradient updates to z *during generation*,
            # then it should be outside no_grad(). However, if it's just a 'push' based on initial Z,
            # then keep it in no_grad().
            # Assuming it's a 'push' based on initial Z for inference, keep in no_grad for now.
            # If this becomes too slow or behaves unexpectedly, review if autograd.grad is truly needed here
            # or if a simpler repulsive force application suffices.

            # For a pure inference-time generation, the 'diversity_loss' part that modifies z
            # with gradients (`z_grad = torch.autograd.grad(repulsion.sum(), z, retain_graph=True)[0]`)
            # is typically not run. If you want a "post-processing" diversity enhancement,
            # you might apply a fixed "push" without needing gradients.
            # For demonstration, I'll keep the original logic, assuming the intent is to show
            # how this 'diversity' idea could influence generation.

            # This part should be re-evaluated if it's meant to be part of an inference API.
            # For training, it makes sense. For generation, usually you just sample Z.

            # If you want this to apply a "repulsion" effect without needing actual gradient steps during inference:
            # You would need to manually compute the repulsion vector and add it to z.
            # For now, sticking to original code.

            # Note: The original code in `EnhancedConvVAE` for `generate_with_diversity_loss`
            # temporarily enables gradients on `z` for the `autograd.grad` call.
            # While this can work inside `no_grad` if `retain_graph=True` is used and `z` initially needs grad,
            # it's unconventional for pure inference. Let's make sure it's handled.

            # Make z require gradients temporarily to compute z_grad
            original_z_requires_grad = z.requires_grad
            z.requires_grad_(True)

            z_expanded = z.unsqueeze(1).expand(-1, num_samples, -1)
            z_t_expanded = z.unsqueeze(0).expand(num_samples, -1, -1)
            distances = torch.norm(z_expanded - z_t_expanded, dim=2)
            diversity_mask = (distances < 1.0) & (distances > 0)

            if diversity_mask.sum() > 0:
                repulsion = torch.where(diversity_mask, 1.0 / (distances + 1e-8), torch.tensor(0.0, device=self.device))
                # sum() over repulsion results in a scalar, which can be backpropagated to z.
                # Since we are in no_grad, we need to explicitly enable gradients on z for this specific operation.
                z_grad = torch.autograd.grad(repulsion.sum(), z, retain_graph=True)[0]
                z = z + diversity_weight * z_grad

            # Restore original requires_grad state
            z.requires_grad_(original_z_requires_grad)

            generated = self.decode(z)
        return generated

    def interpolate_spherical(self, z1: torch.Tensor, z2: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """Spherical interpolation for better results"""
        # Ensure z1 and z2 are 2D tensors (batch_size, latent_dim)
        if z1.dim() == 1:
            z1 = z1.unsqueeze(0)
        if z2.dim() == 1:
            z2 = z2.unsqueeze(0)

        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)

        dot = torch.sum(z1_norm * z2_norm, dim=1, keepdim=True)
        dot = torch.clamp(dot, -1, 1)
        theta = torch.acos(dot)

        t_values = torch.linspace(0, 1, steps, device=z1.device).view(steps, 1, 1)  # Reshape for broadcasting

        sin_theta = torch.sin(theta)

        # Handle case where vectors are parallel
        parallel_mask = sin_theta.abs() < 1e-6

        # Expand for broadcasting
        theta_expanded = theta.unsqueeze(0)
        sin_theta_expanded = sin_theta.unsqueeze(0)
        z1_norm_expanded = z1_norm.unsqueeze(0)
        z2_norm_expanded = z2_norm.unsqueeze(0)

        interp_norm = torch.zeros(steps, z1.size(0), self.latent_dim, device=z1.device)

        if parallel_mask.any():
            # Linear interpolation for parallel vectors
            interp_norm[..., parallel_mask.squeeze()] = (1 - t_values) * z1_norm_expanded[
                ..., parallel_mask.squeeze()] + \
                                                        t_values * z2_norm_expanded[..., parallel_mask.squeeze()]

        # Spherical interpolation for non-parallel vectors
        non_parallel_mask = ~parallel_mask
        if non_parallel_mask.any():
            interp_norm[..., non_parallel_mask.squeeze()] = \
                (torch.sin((1 - t_values) * theta_expanded[..., non_parallel_mask.squeeze()]) / sin_theta_expanded[
                    ..., non_parallel_mask.squeeze()]) * z1_norm_expanded[..., non_parallel_mask.squeeze()] + \
                (torch.sin(t_values * theta_expanded[..., non_parallel_mask.squeeze()]) / sin_theta_expanded[
                    ..., non_parallel_mask.squeeze()]) * z2_norm_expanded[..., non_parallel_mask.squeeze()]

        # Interpolate magnitudes
        z1_mag = torch.norm(z1, dim=1, keepdim=True)
        z2_mag = torch.norm(z2, dim=1, keepdim=True)
        interp_mag = (1 - t_values) * z1_mag.unsqueeze(0) + t_values * z2_mag.unsqueeze(0)

        # Apply magnitude and reshape for decoding
        interp_z = F.normalize(interp_norm, dim=-1) * interp_mag
        interp_z = interp_z.view(-1, self.latent_dim)  # Flatten batch and steps

        with torch.no_grad():
            return self.decode(interp_z)

    def _move_to_device(self) -> None:
        """Move all modules to the specified device"""
        self.to(self.device)

    def to(self, device: torch.device) -> 'EnhancedConvVAE':
        """Override to method to update internal device reference"""
        self.device = device
        super().to(device)
        return self

    def cuda(self, device: Optional[int] = None) -> 'EnhancedConvVAE':
        """Override cuda method to update internal device reference"""
        if device is None:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device(f'cuda:{device}')
        super().cuda(device)
        return self

    def cpu(self) -> 'EnhancedConvVAE':
        """Override cpu method to update internal device reference"""
        self.device = torch.device('cpu')
        super().cpu()
        return self

    def save_model(self, path: str) -> None:
        """Save model state dictionary and hyperparameters"""
        torch.save({
            'state_dict': self.state_dict(),
            'latent_dim': self.latent_dim,
            'base_channels': self.base_channels,
            'image_size': self.image_size,
            'input_channels': self.input_channels,
            'beta': self.beta,
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'use_spectral_norm': self.use_spectral_norm
        }, path)
        print(f"EnhancedConvVAE model saved to {path}")

    def load_model(self, path: str, map_location: Optional[str] = None) -> None:
        """Load model state dictionary and rebuild model if hyperparameters are present"""
        if map_location is None:
            map_location = self.device

        checkpoint = torch.load(path, map_location=map_location)

        # Load hyperparameters if available and rebuild model
        if all(k in checkpoint for k in ['latent_dim', 'base_channels', 'image_size', 'input_channels']):
            self.latent_dim = checkpoint['latent_dim']
            self.base_channels = checkpoint['base_channels']
            self.image_size = checkpoint['image_size']
            self.input_channels = checkpoint['input_channels']
            self.beta = checkpoint.get('beta', 0.5)
            self.gamma = checkpoint.get('gamma', 1000)
            self.learning_rate = checkpoint.get('learning_rate', 1e-4)
            self.use_spectral_norm = checkpoint.get('use_spectral_norm', True)

            # Rebuild model with loaded parameters
            self.build_model()
            print("Model hyperparameters loaded and model rebuilt.")
        else:
            print("Hyperparameters not found in checkpoint. Loading state_dict into existing model.")
            # If no hyperparameters, assume current model structure matches saved state_dict
            # and only load state_dict.

        # Load state dict
        state_dict = checkpoint.get('state_dict', checkpoint)
        self.load_state_dict(state_dict)

        self._move_to_device()
        print(f"EnhancedConvVAE model loaded from {path} and moved to {self.device}")

    def save_generated_images(self, images: torch.Tensor, folder_path: str, prefix: str = "generated_image",
                              normalize: bool = True) -> None:
        """Save a batch of generated images as individual PNG files"""
        os.makedirs(folder_path, exist_ok=True)
        for i, img in enumerate(images):
            if normalize:
                # Assuming images are in [-1, 1] and need to be normalized to [0, 1]
                img = (img + 1) / 2
            filepath = os.path.join(folder_path, f"{prefix}_{i:04d}.png")
            save_image(img.cpu(), filepath)
        print(f"Saved {len(images)} generated images to {folder_path}")


# --- Example Usage in Jupyter Notebook ---

if __name__ == "__main__":
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Preparation (using CIFAR-10 as an example)
    image_size = 64
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # 3. Model Initialization
    vae = EnhancedConvVAE(
        device=device,
        input_channels=3,
        latent_dim=256,
        base_channels=64,
        image_size=image_size,
        learning_rate=1e-4,
        beta=0.5,
        gamma=1000,
        use_spectral_norm=True
    )

    # 4. Training the Model
    num_epochs = 10  # You can set this higher for better results
    print("\nTraining EnhancedConvVAE...")
    training_history = vae.train_architecture(train_loader, epochs=num_epochs, use_capacity=True)



    # 6. Evaluate the Model (Optional)
    print("\nEvaluating EnhancedConvVAE on test set...")
    eval_metrics = vae.evaluate(test_loader)
    print(f"Test Set Metrics: "
          f"Total Loss: {eval_metrics['total_loss']:.4f}, "
          f"Recon Loss: {eval_metrics['reconstruction_loss']:.4f}, "
          f"KL Loss: {eval_metrics['kl_loss']:.4f}, "
          f"MAE: {eval_metrics['mae']:.4f}, "
          f"PSNR: {eval_metrics['psnr']:.2f} dB")

    # 7. Generate and Visualize Images
    vae.eval()  # Set to evaluation mode

    # Generate diverse samples
    num_samples_gen = 25
    print("\nGenerating diverse samples...")
    diverse_samples = vae.generate_diverse(num_samples_gen, temperature=1.2)
    vae.save_generated_images(diverse_samples, folder_path='./generated_images_diverse', prefix='diverse_sample')


    # Generate samples with diversity loss encouragement
    print("\nGenerating samples with diversity loss encouragement...")
    diversity_loss_samples = vae.generate_with_diversity_loss(num_samples_gen, diversity_weight=0.2)
    vae.save_generated_images(diversity_loss_samples, folder_path='./generated_images_diversity_loss',
                              prefix='diversity_loss_sample')

    # 8. Visualize Reconstruction
    print("\nVisualizing reconstructions...")
    vae.eval()
    data_iter = iter(test_loader)
    original_batch, _ = next(data_iter)
    original_batch = original_batch.to(device)

    num_recons_to_show = 8
    with torch.no_grad():
        reconstructed_batch, _, _ = vae(original_batch[:num_recons_to_show])



    # 9. Visualize Spherical Interpolation
    print("\nVisualizing spherical interpolation...")
    # Get two random images from the test set to interpolate between
    idx1, idx2 = np.random.choice(len(test_dataset), 2, replace=False)
    img_a = transform(test_dataset.data[idx1]).unsqueeze(0).to(device)
    img_b = transform(test_dataset.data[idx2]).unsqueeze(0).to(device)

    with torch.no_grad():
        mu_a, _ = vae.encode(img_a)
        mu_b, _ = vae.encode(img_b)
        interpolated_imgs = vae.interpolate_spherical(mu_a, mu_b, steps=15)


    # 10. Save and Load Model
    model_save_path = './enhanced_vae_checkpoint.pth'
    vae.save_model(model_save_path)

    # Example of loading the model
    # vae_loaded = EnhancedConvVAE(device=device, input_channels=3, image_size=image_size) # Basic init
    # vae_loaded.load_model(model_save_path)
    # print("Model successfully loaded and rebuilt.")