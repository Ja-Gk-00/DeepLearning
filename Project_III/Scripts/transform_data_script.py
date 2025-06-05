#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Supported image extensions
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)


def find_image_files(root_dir):
    """Recursively find all image files under root_dir."""
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if is_image_file(fname):
                yield Path(dirpath) / fname


def classify_label(filename):
    """Return 'cat' if 'cat' in name, 'dog' if 'dog' in name, else None."""
    name = filename.stem.lower()
    if "cat" in name:
        return "cat"
    if "dog" in name:
        return "dog"
    return None


def transform_and_save(img_path, out_path, transform, save_format):
    """
    Load image, apply transform, and save as specified format.
    - save_format 'pt': torch.save
    - save_format 'npy': numpy.save
    """
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img)
    try:
        if save_format == "pt":
            # Save as PyTorch tensor
            torch.save(tensor, out_path)
        else:
            # Save as NumPy array
            arr = tensor.numpy()
            np.save(out_path.with_suffix(".npy"), arr)
    except Exception as e:
        print(f"[ERROR] Failed to save {out_path} ({save_format}): {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Transform images into PyTorch tensors or NumPy arrays and optionally order by class."
    )
    parser.add_argument(
        "--input_dir", "-i", required=True, help="Directory with raw image files."
    )
    parser.add_argument(
        "--output_dir", "-o", required=True, help="Directory to save transformed data."
    )
    parser.add_argument(
        "--order", "-r", action="store_true",
        help="If set, classify files into 'cat' or 'dog' subdirectories based on filename."
    )
    parser.add_argument(
        "--format", "-f", choices=["pt", "npy"], default="pt",
        help="Save format: 'pt' for PyTorch tensors, 'npy' for NumPy arrays."
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define basic transform: PIL Image -> Tensor (C x H x W), pixels in [0,1]
    transform = transforms.ToTensor()

    for img_path in tqdm(list(find_image_files(input_dir)), desc="Processing images"):
        label = classify_label(img_path)
        if args.order:
            # Skip files that are neither cat nor dog
            if label is None:
                continue
            class_dir = output_dir / label
            class_dir.mkdir(parents=True, exist_ok=True)
            base_name = img_path.stem
            if args.format == "pt":
                out_file = class_dir / (base_name + ".pt")
            else:
                out_file = class_dir / (base_name + ".npy")
        else:
            base_name = img_path.stem
            if args.format == "pt":
                out_file = output_dir / (base_name + ".pt")
            else:
                out_file = output_dir / (base_name + ".npy")

        transform_and_save(img_path, out_file, transform, args.format)

    print("Data transformation complete. Transformed files saved to:", output_dir)


if __name__ == "__main__":
    main()
