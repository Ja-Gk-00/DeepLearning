#!/usr/bin/env python3
import os
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Union
from torch.nn.functional import interpolate
from tqdm import tqdm


class ImageDownscaler:
    """
    A class to downscale images in a folder to match the lowest resolution found in the dataset.
    Works with .pt tensor files and is compatible with the existing DataLoader class.
    """

    def __init__(self, data_dir: Union[str, Path], output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the ImageDownscaler.

        Args:
            data_dir: Path to the folder containing .pt image files
            output_dir: Path to save downscaled images. If None, overwrites original files.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir
        self.min_height = None
        self.min_width = None
        self.image_files = []

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Find all .pt files
        self._find_image_files()

        if not self.image_files:
            raise ValueError(f"No .pt files found in {self.data_dir}")

    def _find_image_files(self) -> None:
        """Find all .pt files in the data directory."""
        self.image_files = list(self.data_dir.rglob("*.pt"))
        print(f"Found {len(self.image_files)} .pt files")

    def _get_tensor_dimensions(self, tensor_path: Path) -> Tuple[int, int, int]:
        """
        Get the dimensions of a tensor file.

        Args:
            tensor_path: Path to the .pt file

        Returns:
            Tuple of (channels, height, width)
        """
        try:
            tensor = torch.load(tensor_path, map_location='cpu')

            # Handle different tensor shapes
            if tensor.ndim == 3:  # [C, H, W]
                channels, height, width = tensor.shape
            elif tensor.ndim == 4:  # [N, C, H, W] - take first image
                _, channels, height, width = tensor.shape
            else:
                raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

            return channels, height, width

        except Exception as e:
            print(f"Error loading {tensor_path}: {e}")
            return None

    def find_minimum_resolution(self) -> Tuple[int, int]:
        """
        Find the minimum resolution (height, width) among all images.

        Returns:
            Tuple of (min_height, min_width)
        """
        print("Scanning images to find minimum resolution...")

        min_height = float('inf')
        min_width = float('inf')
        valid_files = []

        for img_path in tqdm(self.image_files, desc="Analyzing images"):
            dims = self._get_tensor_dimensions(img_path)
            if dims is not None:
                channels, height, width = dims
                min_height = min(min_height, height)
                min_width = min(min_width, width)
                valid_files.append(img_path)

        if not valid_files:
            raise ValueError("No valid image files found")

        # Update the list to only include valid files
        self.image_files = valid_files

        self.min_height = int(min_height)
        self.min_width = int(min_width)

        print(f"Minimum resolution found: {self.min_height}x{self.min_width}")
        print(f"Valid files: {len(self.image_files)}")

        return self.min_height, self.min_width

    def downscale_image(self, tensor: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """
        Downscale a single image tensor to target dimensions.

        Args:
            tensor: Input tensor of shape [C, H, W] or [N, C, H, W]
            target_height: Target height
            target_width: Target width

        Returns:
            Downscaled tensor
        """
        original_shape = tensor.shape

        # Handle different input shapes
        if tensor.ndim == 3:  # [C, H, W]
            # Add batch dimension for interpolation
            tensor_batch = tensor.unsqueeze(0)  # [1, C, H, W]
            process_batch = True
        elif tensor.ndim == 4:  # [N, C, H, W]
            tensor_batch = tensor
            process_batch = False
        else:
            raise ValueError(f"Unsupported tensor shape: {original_shape}")

        # Perform interpolation
        downscaled = interpolate(
            tensor_batch,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        )

        # Remove batch dimension if it was added
        if process_batch:
            downscaled = downscaled.squeeze(0)

        return downscaled

    def process_single_image(self, img_path: Path, target_height: int, target_width: int) -> bool:
        """
        Process a single image file.

        Args:
            img_path: Path to the image file
            target_height: Target height for downscaling
            target_width: Target width for downscaling

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the tensor
            tensor = torch.load(img_path, map_location='cpu')

            # Get current dimensions
            if tensor.ndim == 3:
                _, current_height, current_width = tensor.shape
            elif tensor.ndim == 4:
                _, _, current_height, current_width = tensor.shape
            else:
                print(f"Skipping {img_path}: Unexpected tensor shape {tensor.shape}")
                return False

            # Skip if already at target resolution
            if current_height == target_height and current_width == target_width:
                # If output directory is different, copy the file
                if self.output_dir != self.data_dir:
                    output_path = self.output_dir / img_path.relative_to(self.data_dir)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(tensor, output_path)
                return True

            # Downscale the image
            downscaled_tensor = self.downscale_image(tensor, target_height, target_width)

            # Determine output path
            if self.output_dir != self.data_dir:
                output_path = self.output_dir / img_path.relative_to(self.data_dir)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_path = img_path

            # Save the downscaled tensor
            torch.save(downscaled_tensor, output_path)

            return True

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return False

    def downscale_all_images(self, target_height: Optional[int] = None, target_width: Optional[int] = None) -> None:
        """
        Downscale all images to the specified or minimum resolution.

        Args:
            target_height: Target height. If None, uses minimum found height.
            target_width: Target width. If None, uses minimum found width.
        """
        # Find minimum resolution if not provided
        if target_height is None or target_width is None:
            min_h, min_w = self.find_minimum_resolution()
            target_height = target_height or min_h
            target_width = target_width or min_w

        print(f"Downscaling all images to {target_height}x{target_width}")

        successful = 0
        failed = 0

        for img_path in tqdm(self.image_files, desc="Downscaling images"):
            if self.process_single_image(img_path, target_height, target_width):
                successful += 1
            else:
                failed += 1

        print(f"Downscaling completed:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Output directory: {self.output_dir}")

    def get_statistics(self) -> dict:
        """
        Get statistics about the images in the dataset.

        Returns:
            Dictionary with statistics
        """
        if not self.image_files:
            return {}

        print("Gathering dataset statistics...")

        resolutions = []
        channels_list = []
        file_sizes = []

        for img_path in tqdm(self.image_files, desc="Analyzing images"):
            dims = self._get_tensor_dimensions(img_path)
            if dims is not None:
                channels, height, width = dims
                resolutions.append((height, width))
                channels_list.append(channels)
                file_sizes.append(img_path.stat().st_size)

        if not resolutions:
            return {}

        heights, widths = zip(*resolutions)

        stats = {
            'total_files': len(self.image_files),
            'min_resolution': (min(heights), min(widths)),
            'max_resolution': (max(heights), max(widths)),
            'unique_resolutions': len(set(resolutions)),
            'channels': {
                'min': min(channels_list),
                'max': max(channels_list),
                'unique': list(set(channels_list))
            },
            'file_sizes_mb': {
                'min': min(file_sizes) / (1024 * 1024),
                'max': max(file_sizes) / (1024 * 1024),
                'avg': sum(file_sizes) / len(file_sizes) / (1024 * 1024)
            }
        }

        return stats

    def print_statistics(self) -> None:
        """Print dataset statistics."""
        stats = self.get_statistics()

        if not stats:
            print("No statistics available")
            return

        print("\n" + "=" * 50)
        print("DATASET STATISTICS")
        print("=" * 50)
        print(f"Total files: {stats['total_files']}")
        print(f"Minimum resolution: {stats['min_resolution'][0]}x{stats['min_resolution'][1]}")
        print(f"Maximum resolution: {stats['max_resolution'][0]}x{stats['max_resolution'][1]}")
        print(f"Unique resolutions: {stats['unique_resolutions']}")
        print(f"Channels: {stats['channels']['unique']}")
        print(
            f"File sizes (MB): {stats['file_sizes_mb']['min']:.2f} - {stats['file_sizes_mb']['max']:.2f} (avg: {stats['file_sizes_mb']['avg']:.2f})")
        print("=" * 50 + "\n")


# Example usage and integration with existing classes
if __name__ == "__main__":
    # Get the current directory of the script (DataObjects)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up one level to the project root
    project_root = os.path.dirname(current_script_dir)

    input_directory = os.path.join(project_root, 'data', 'transformed')
    output_directory = os.path.join(project_root, 'data', 'downscaled', 'cats')

    # Downscale to different folder (preserves original files)
    downscaler = ImageDownscaler(input_directory, output_directory)

    # Print statistics about the dataset
    downscaler.print_statistics()

    # Downscale all images to minimum resolution
    downscaler.downscale_all_images()

    # # Now you can use the downscaled images with your existing DataLoader
    # from DataObjects import DataLoader
    # from Architectures.ConvolutionalAutoEncoder import ConvAutoEncoder
    #
    # # Create DataLoader with downscaled images
    # data_loader = DataLoader(
    #     data_dir=output_folder,  # or data_folder if you overwrote original files
    #     batch_size=32,
    #     shuffle=True,
    #     fraction=1.0
    # )
    #
    # # Initialize autoencoder with the minimum resolution found
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # # Get the minimum resolution for autoencoder initialization
    # min_height, min_width = downscaler.min_height, downscaler.min_width
    # image_size = min(min_height, min_width)  # Assuming square images for the autoencoder
    #
    # autoencoder = ConvAutoEncoder(
    #     device=device,
    #     input_channels=3,  # Adjust based on your images
    #     latent_dim=128,
    #     base_channels=64,
    #     image_size=image_size,
    #     learning_rate=1e-3
    # )
    #
    # # Configure optimizers
    # autoencoder.configure_optimizers()
    #
    # # Train the autoencoder
    # print(f"Training autoencoder with image size: {image_size}x{image_size}")
    # autoencoder.train_architecture(data_loader)