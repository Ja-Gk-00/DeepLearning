# VAE Diversity Enhancement - Jupyter Notebook Usage
# Run this in your separate Jupyter notebook

import torch
import torch.nn.functional as F
import torchvision.utils
import numpy as np
import os
from Architectures.ConvolutionalAutoEncoder import ConvVAE


# Make sure to import your ConvVAE class
# from your_vae_file import ConvVAE  # Adjust import path as needed

# ==== DIVERSITY FUNCTIONS ====
def generate_diverse_samples(vae_model, num_samples=16):
    """Generate diverse samples using different sampling techniques"""
    vae_model.eval()

    # Method 1: Higher temperature sampling
    high_temp_samples = vae_model.generate(num_samples=num_samples // 4, temperature=2.0)

    # Method 2: Spherical interpolation in latent space
    z1 = vae_model.sample_latent(1, temperature=1.5)
    z2 = vae_model.sample_latent(1, temperature=1.5)

    # Spherical interpolation (slerp)
    def slerp(z1, z2, steps):
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)

        dot = torch.sum(z1_norm * z2_norm, dim=1, keepdim=True)
        theta = torch.acos(torch.clamp(dot, -1, 1))

        t_values = torch.linspace(0, 1, steps, device=z1.device).view(-1, 1)

        sin_theta = torch.sin(theta)
        interp_z = (torch.sin((1 - t_values) * theta) / sin_theta) * z1 + \
                   (torch.sin(t_values * theta) / sin_theta) * z2

        z1_mag = torch.norm(z1, dim=1, keepdim=True)
        z2_mag = torch.norm(z2, dim=1, keepdim=True)
        interp_mag = (1 - t_values) * z1_mag + t_values * z2_mag

        return F.normalize(interp_z, dim=1) * interp_mag

    slerp_z = slerp(z1, z2, num_samples // 4)
    slerp_samples = vae_model.decode(slerp_z)

    # Method 3: Structured latent sampling
    structured_samples = []
    for i in range(num_samples // 4):
        z = torch.randn(1, vae_model.latent_dim, device=vae_model.device)
        extreme_dims = torch.randperm(vae_model.latent_dim)[:vae_model.latent_dim // 4]
        z[0, extreme_dims] = torch.sign(z[0, extreme_dims]) * (
                    2 + torch.rand(len(extreme_dims), device=vae_model.device))
        structured_samples.append(vae_model.decode(z))

    structured_samples = torch.cat(structured_samples, dim=0)

    # Method 4: Normal temperature
    normal_samples = vae_model.generate(num_samples=num_samples // 4, temperature=1.0)

    # Combine all methods
    all_samples = torch.cat([high_temp_samples, slerp_samples, structured_samples, normal_samples], dim=0)
    return all_samples


def generate_with_constraints(vae_model, num_samples=16):
    """Generate samples with explicit diversity constraints"""
    vae_model.eval()
    samples = []
    used_latents = []

    max_attempts = num_samples * 3

    for _ in range(max_attempts):
        z = vae_model.sample_latent(1, temperature=1.2)

        is_diverse = True
        for used_z in used_latents:
            distance = torch.norm(z - used_z, p=2)
            if distance < 1.0:
                is_diverse = False
                break

        if is_diverse or len(used_latents) == 0:
            sample = vae_model.decode(z)
            samples.append(sample)
            used_latents.append(z)

            if len(samples) >= num_samples:
                break

    return torch.cat(samples, dim=0)


def explore_latent_space(vae_model, num_samples=16):
    """Systematically explore different regions of latent space"""
    vae_model.eval()
    samples = []

    sqrt_samples = int(np.sqrt(num_samples))

    for i in range(sqrt_samples):
        for j in range(sqrt_samples):
            z = torch.randn(1, vae_model.latent_dim, device=vae_model.device) * 0.5

            offset = torch.zeros_like(z)

            if vae_model.latent_dim >= 2:
                offset[0, 0] = (i / sqrt_samples - 0.5) * 4
                offset[0, 1] = (j / sqrt_samples - 0.5) * 4

            z_final = z + offset
            sample = vae_model.decode(z_final)
            samples.append(sample)

    return torch.cat(samples, dim=0)


# ==== MAIN USAGE ====

# 1. Load your trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model with same parameters as training
vae = ConvVAE.from_pretrained(
    device=device,
    input_channels=3,
    latent_dim=128,  # Make sure this matches your trained model
    base_channels=64,
    image_size=64,
    learning_rate=1e-3,
    beta=1.0
)

# Load your trained weights
vae.load_model("path/to/your/trained_model.pth")  # Replace with actual path

print(f"Model loaded on device: {device}")

# 2. Generate diverse samples using different methods

# Method 1: Simple high temperature
print("Generating high temperature samples...")
high_temp_samples = vae.generate(num_samples=16, temperature=2.5)

# Method 2: Advanced diverse sampling
print("Generating diverse samples...")
diverse_samples = generate_diverse_samples(vae, num_samples=16)

# Method 3: Constrained generation
print("Generating constrained samples...")
constrained_samples = generate_with_constraints(vae, num_samples=16)

# Method 4: Latent space exploration
print("Exploring latent space...")
exploration_samples = explore_latent_space(vae, num_samples=16)

# 3. Save all results
os.makedirs("../Saved_Models", exist_ok=True)

methods = {
    'high_temperature': high_temp_samples,
    'diverse_sampling': diverse_samples,
    'constrained_generation': constrained_samples,
    'latent_exploration': exploration_samples
}

for method_name, samples in methods.items():
    # Normalize from [-1, 1] to [0, 1]
    normalized = (samples + 1) / 2

    # Save as grid
    torchvision.utils.save_image(
        normalized.cpu(),
        f"../Saved_Models/{method_name}_samples.png",
        nrow=4,
        normalize=False  # Already normalized
    )

    print(f"Saved {method_name} samples to ../Saved_Models/{method_name}_samples.png")

print("All samples generated and saved!")

# 4. Optional: Compare with original generation
print("\nComparing with original generation...")
original_samples = vae.generate(num_samples=16, temperature=1.0)
normalized_original = (original_samples + 1) / 2

torchvision.utils.save_image(
    normalized_original.cpu(),
    "../Saved_Models/original_samples.png",
    nrow=4
)

print("Original samples saved for comparison.")


# 5. Optional: Quick diversity check
def calculate_diversity_metric(samples):
    """Calculate a simple diversity metric"""
    with torch.no_grad():
        # Flatten images
        flat_samples = samples.view(samples.size(0), -1)

        # Calculate pairwise distances
        distances = []
        for i in range(len(flat_samples)):
            for j in range(i + 1, len(flat_samples)):
                dist = torch.norm(flat_samples[i] - flat_samples[j], p=2)
                distances.append(dist.item())

        return np.mean(distances), np.std(distances)


print("\nDiversity metrics (higher = more diverse):")
for method_name, samples in methods.items():
    mean_dist, std_dist = calculate_diversity_metric(samples)
    print(f"{method_name}: mean={mean_dist:.3f}, std={std_dist:.3f}")

# Compare with original
orig_mean, orig_std = calculate_diversity_metric(original_samples)
print(f"original: mean={orig_mean:.3f}, std={orig_std:.3f}")