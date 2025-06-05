#!/usr/bin/env python3
import argparse
import os
import zipfile
import shutil
from tqdm import tqdm

# Attempt to import necessary libraries
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

# Kaggle competition identifier for Dogs vs. Cats
DOGS_VS_CATS_COMPETITION = "dogs-vs-cats"


def unzip_file(zip_path, extract_to):
    """Unzip a .zip archive to a target directory."""
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in tqdm(z.infolist(), desc=f"Extracting {os.path.basename(zip_path)}"):
            z.extract(member, extract_to)


def download_primary_dataset(output_dir):
    """Download and prepare the primary dataset via kagglehub."""
    if not KAGGLEHUB_AVAILABLE:
        raise RuntimeError("kagglehub is not installed. Run 'pip install kagglehub' first.")
    # Download returns path to downloaded file or folder
    print("Downloading primary dataset (borhanitrash/cat-dataset)...")
    dataset_path = kagglehub.dataset_download("borhanitrash/cat-dataset")
    print("Path to dataset files:", dataset_path)

    extract_dir = os.path.join(output_dir, "primary_data")
    os.makedirs(extract_dir, exist_ok=True)

    # If the returned path is a zip file, unzip it
    if dataset_path.endswith(".zip"):
        unzip_file(dataset_path, extract_dir)
        os.remove(dataset_path)
        print("Primary data extracted to:", extract_dir)
    else:
        # If it's already a folder, move its contents into output_dir/primary_data
        if os.path.isdir(dataset_path):
            for item in os.listdir(dataset_path):
                src = os.path.join(dataset_path, item)
                dst = os.path.join(extract_dir, item)
                shutil.move(src, dst)
            print("Primary data moved to:", extract_dir)
        else:
            raise RuntimeError(f"Unexpected dataset_path: {dataset_path}")


def download_dogs_vs_cats(output_dir):
    """Download Dogs vs. Cats via Kaggle API, unzip, and organize."""
    if not KAGGLE_AVAILABLE:
        raise RuntimeError("Kaggle API is not installed. Run 'pip install kaggle' and configure credentials.")
    api = KaggleApi()
    api.authenticate()

    zip_dest = os.path.join(output_dir, "dogs-vs-cats", "train.zip")
    os.makedirs(os.path.dirname(zip_dest), exist_ok=True)

    print("Downloading Dogs vs. Cats (train.zip)...")
    api.competition_download_file(
        DOGS_VS_CATS_COMPETITION, file_name="train.zip", path=os.path.dirname(zip_dest), force=False
    )

    extract_dir = os.path.join(output_dir, "dogs-vs-cats", "train")
    os.makedirs(extract_dir, exist_ok=True)
    unzip_file(zip_dest, extract_dir)
    os.remove(zip_dest)
    print("Dogs vs. Cats data ready at:", extract_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare image datasets: primary (borhanitrash/cat-dataset) and optional Dogs vs. Cats."
    )
    parser.add_argument(
        "--output_dir", "-o", required=True, help="Directory where all data will be stored."
    )
    parser.add_argument(
        "--download_dogs_cats", "-d", action="store_true",
        help="If set, also download and prepare the Dogs vs. Cats dataset."
    )
    args = parser.parse_args()

    # download primary dataset via kagglehub
    download_primary_dataset(args.output_dir)

    # optionally download Dogs vs. Cats
    if args.download_dogs_cats:
        download_dogs_vs_cats(args.output_dir)


if __name__ == "__main__":
    main()
