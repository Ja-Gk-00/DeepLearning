#!/usr/bin/env python3
import os
import argparse
import yaml
import logging
import random
import numpy as np
from PIL import Image, ImageEnhance
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# Global dictionary to register augmentation functions.
AUGMENTATIONS = {}

def register_augmentation(name: str):
    def decorator(func):
        AUGMENTATIONS[name] = func
        return func
    return decorator

# === SIMPLE AUGMENTATIONS ===

@register_augmentation("rotation")
def augment_rotation(image: Image.Image, angle: float = 30) -> Image.Image:
    return image.rotate(angle)

@register_augmentation("horizontal_flip")
def augment_horizontal_flip(image: Image.Image) -> Image.Image:
    return image.transpose(Image.FLIP_LEFT_RIGHT)

@register_augmentation("gaussian_noise")
def augment_gaussian_noise(image: Image.Image, mean: float = 0.0, std: float = 10.0) -> Image.Image:
    arr = np.array(image).astype(np.float32)
    noise = np.random.normal(mean, std, arr.shape)
    arr = arr + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

@register_augmentation("brightness")
def augment_brightness(image: Image.Image, factor: float = 1.5) -> Image.Image:
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

# === ADVANCED AUGMENTATIONS ===

@register_augmentation("autoaugment")
def augment_autoaugment(image: Image.Image) -> Image.Image:
    transform = AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
    return transform(image)

@register_augmentation("cutmix")
def augment_cutmix(image: Image.Image, other_image: Image.Image, beta: float = 1.0) -> Image.Image:
    if image.mode != "RGB" or other_image.mode != "RGB":
        image = image.convert("RGB")
        other_image = other_image.convert("RGB")
        
    arr1 = np.array(image)
    arr2 = np.array(other_image)
    
    if arr1.ndim != arr2.ndim:
        if arr1.ndim == 2:
            arr1 = np.stack([arr1]*3, axis=-1)
        elif arr2.ndim == 2:
            arr2 = np.stack([arr2]*3, axis=-1)
            
    h, w = arr1.shape[0], arr1.shape[1]
    lam = np.random.beta(beta, beta)
    cut_rat = np.sqrt(1 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    
    arr1[y1:y2, x1:x2] = arr2[y1:y2, x1:x2]
    return Image.fromarray(arr1)

# === CUSTOM TILE SWAP AUGMENTATION ===

@register_augmentation("tileswap")
def augment_tileswap(image: Image.Image, n: int = 2, strategy: str = "random", m: int = 1) -> Image.Image:
    arr = np.array(image)
    h, w = arr.shape[0], arr.shape[1]
    tile_h = h // n
    tile_w = w // n
    tiles = [(i, j) for i in range(n) for j in range(n)]
    tile_list = []
    for i in range(n):
        row_tiles = []
        for j in range(n):
            tile = arr[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w].copy()
            row_tiles.append(tile)
        tile_list.append(row_tiles)
    for _ in range(m):
        if strategy == "random":
            pos1 = random.choice(tiles)
            pos2 = random.choice(tiles)
            while pos2 == pos1:
                pos2 = random.choice(tiles)
        elif strategy == "neighbor":
            pos1 = random.choice(tiles)
            i, j = pos1
            neighbors = []
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < n and 0 <= nj < n:
                    neighbors.append((ni, nj))
            pos2 = random.choice(neighbors) if neighbors else pos1
        else:
            continue
        i1, j1 = pos1
        i2, j2 = pos2
        tile_list[i1][j1], tile_list[i2][j2] = tile_list[i2][j2], tile_list[i1][j1]
    new_arr = np.zeros_like(arr)
    for i in range(n):
        for j in range(n):
            new_arr[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w] = tile_list[i][j]
    return Image.fromarray(new_arr)

# === HELPER FUNCTIONS FOR IMAGE I/O ===

def load_image(file_path: str, input_format: str) -> Image.Image:
    if input_format == "png":
        return Image.open(file_path)
    elif input_format == "npy":
        arr = np.load(file_path)
        return Image.fromarray(arr.astype(np.uint8), mode='L')
    else:
        raise ValueError(f"Unsupported input_format: {input_format}")

def save_image(image: Image.Image, save_path: str, output_format: str) -> None:
    if output_format == "png":
        image.save(save_path)
    elif output_format == "npy":
        arr = np.array(image)
        np.save(save_path, arr)
    else:
        raise ValueError(f"Unsupported output_format: {output_format}")

# === PROCESSING FUNCTIONS ===

def process_file(file_path: str, input_dir: str, output_dir: str, prefix: str, config: dict, input_format: str, output_format: str) -> None:
    rel_path = os.path.relpath(file_path, input_dir)
    file_dir, file_name = os.path.split(rel_path)
    
    try:
        portion: float = config.get("portion", 1.0)
        if random.random() > portion:
            return

        image = load_image(file_path, input_format)
        aug_name = config.get("augmentation")
        args = config.get("args", {})

        if aug_name not in AUGMENTATIONS:
            logging.warning(f"Augmentation '{aug_name}' not recognized. Skipping file {file_path}.")
            return

        if aug_name == "cutmix":
            dir_path = os.path.dirname(file_path)
            if input_format == "png":
                candidates = [f for f in os.listdir(dir_path) if f.endswith(".png")]
            else:
                candidates = [f for f in os.listdir(dir_path) if f.endswith(".npy")]
            if len(candidates) < 2:
                logging.warning(f"Not enough images for cutmix in {dir_path}. Skipping file {file_path}.")
                return
            candidates = [f for f in candidates if f != os.path.basename(file_path)]
            if not candidates:
                logging.warning(f"No alternate image for cutmix in {dir_path}. Skipping file {file_path}.")
                return
            other_file = os.path.join(dir_path, random.choice(candidates))
            other_image = load_image(other_file, input_format)
            augmented_image = AUGMENTATIONS[aug_name](image, other_image, **args)
        else:
            augmented_image = AUGMENTATIONS[aug_name](image, **args)
    
        new_filename, ext = os.path.splitext(file_name)
        new_filename = f"{prefix}_{aug_name}_{new_filename}{ext}"
        out_dir = os.path.join(output_dir, aug_name, file_dir)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, new_filename)
        save_image(augmented_image, save_path, output_format)
    except Exception as e:
        logging.error(f"Error processing file {file_path} with config {config}: {e}")

def process_directory(input_dir: str, output_dir: str, prefix: str, configs: list, input_format: str, output_format: str) -> None:
    ext = ".png" if input_format == "png" else ".npy"
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(ext):
                file_path = os.path.join(root, file)
                for config in configs:
                    process_file(file_path, input_dir, output_dir, prefix, config, input_format, output_format)

# === MAIN FUNCTION ===

def main():
    parser = argparse.ArgumentParser(description="Data Augmentation Script")
    parser.add_argument("--config_dir", type=str, required=True,
                        help="Directory with YAML configuration files (all will be processed)")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    config_files = [os.path.join(args.config_dir, f) for f in os.listdir(args.config_dir) if f.endswith(".yaml")]
    if not config_files:
        logging.error("No YAML configuration files found in the specified directory.")
        return

    for config_file in config_files:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        global_config = config_data.get("global", {})
        augmentations_list = config_data.get("augmentations", [])
        if not global_config or not augmentations_list:
            logging.warning(f"Configuration file {config_file} is missing 'global' or 'augmentations' keys. Skipping.")
            continue
        
        input_dir = global_config.get("input_dir")
        output_dir = global_config.get("output_dir")
        prefix = global_config.get("prefix", "")
        input_format = global_config.get("input_format", "png")
        output_format = global_config.get("output_format", "png")
        
        if not input_dir or not output_dir:
            logging.warning(f"Configuration file {config_file} does not specify input_dir or output_dir. Skipping.")
            continue
        
        logging.info(f"Processing configuration file: {config_file}")
        for aug_config in augmentations_list:
            logging.info(f"Running augmentation: {aug_config}")
            process_directory(input_dir, output_dir, prefix, [aug_config], input_format, output_format)
    
    logging.info("Data augmentation completed.")

if __name__ == "__main__":
    main()
