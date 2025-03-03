#!/usr/bin/env python3
import os
import sys
import shutil
import traceback
import argparse
from PIL import Image
import numpy as np
from typing import Any

def convert_image(source_file: str, dest_base: str, save_format: str = 'npy', verbose: bool = False) -> None:
    try:
        img: Any = Image.open(source_file)
        img_array: Any = np.array(img)
        if save_format == 'npy':
            np.save(dest_base, img_array)
            saved_path: str = dest_base + ".npy"
        elif save_format == 'binary':
            import pickle
            with open(dest_base + ".pkl", "wb") as f:
                pickle.dump(img_array, f)
            saved_path: str = dest_base + ".pkl"
        else:
            raise ValueError("Unsupported save_format: {}".format(save_format))
        if verbose:
            print(f"Converted: {source_file} -> {saved_path}")
    except Exception as e:
        print(f"Error converting {source_file}: {e}")
        raise

def process_directory(source_dir: str, dest_dir: str, save_format: str = 'npy', verbose: bool = False) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    for item in os.listdir(source_dir):
        source_item: str = os.path.join(source_dir, item)
        dest_item: str = os.path.join(dest_dir, item)
        if os.path.isdir(source_item):
            process_directory(source_item, dest_item, save_format, verbose)
        else:
            if item.lower().endswith('.png'):
                base_name: str = os.path.splitext(item)[0]
                dest_base: str = os.path.join(dest_dir, base_name)
                convert_image(source_item, dest_base, save_format, verbose)
            else:
                shutil.copy2(source_item, dest_item)
                if verbose:
                    print(f"Copied non-image file: {source_item} -> {dest_item}")

def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Convert CINIC10 dataset images to numpy arrays or binary objects while preserving folder structure."
    )
    parser.add_argument(
        '--format', 
        type=str, 
        choices=['npy', 'binary'], 
        default='npy', 
        help="Output format: 'npy' for NumPy arrays or 'binary' for pickle binary objects."
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=os.path.join("Data", "Data_converted"), 
        help="Output directory to save converted data."
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Increase output verbosity."
    )
    args: argparse.Namespace = parser.parse_args()

    source_base: str = os.path.join("Data", "Data_raw")
    dest_base: str = args.output_dir

    try:
        process_directory(source_base, dest_base, args.format, args.verbose)
        if args.verbose:
            print("Data conversion completed successfully.")
    except Exception:
        print("An error occurred during data conversion:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
