#!/usr/bin/env python3
"""
Script to download the TensorFlow Speech Commands dataset and partition it into
training, validation, and testing splits.

Usage:
    python download_and_partition_speech_commands.py \
        --data_dir /path/to/data \
        [--url URL] \
        [--validation_percentage 10] \
        [--testing_percentage 10]
"""
import argparse
import hashlib
import os
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path

def download_dataset(url: str, download_path: Path):
    print(f"Downloading dataset from {url}...")
    urllib.request.urlretrieve(url, download_path)
    print(f"Saved archive to {download_path}")
    print(f"Absolute path saved: {Path.absolute(download_path)}")


def extract_archive(archive_path: Path, extract_to: Path):
    print(f"Extracting {archive_path} to {extract_to}...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)
    print("Extraction complete.")


def which_set(filename: str, validation_percentage: float, testing_percentage: float) -> str:
    base_name = os.path.basename(filename)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # Stable hash based on SHA1
    hash_digest = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
    percentage_hash = ((int(hash_digest, 16) % (2**27 - 1)) * (100.0 / (2**27 - 1)))
    if percentage_hash < validation_percentage:
        return 'validation'
    if percentage_hash < (validation_percentage + testing_percentage):
        return 'testing'
    return 'training'


def partition_dataset(raw_dir: Path, output_dir: Path,
                      validation_percentage: float,
                      testing_percentage: float):
    labels = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    # Ensure output dirs exist
    for split in ['training', 'validation', 'testing']:
        for label in labels:
            # Map background noise files to 'silence'
            target_label = 'silence' if label == '_background_noise_' else label
            (output_dir / split / target_label).mkdir(parents=True, exist_ok=True)

    # Copy files
    for label in labels:
        source_dir = raw_dir / label
        for wav_path in source_dir.glob('*.wav'):
            if label == '_background_noise_':
                # Treat background noise as 'silence' in training only
                split = 'training'
                target_label = 'silence'
            else:
                split = which_set(str(wav_path), validation_percentage, testing_percentage)
                target_label = label
            dest_dir = output_dir / split / target_label
            shutil.copy2(wav_path, dest_dir / wav_path.name)
    print(f"Partitioned dataset into '{output_dir}' with splits: training, validation, testing.")


def main():
    parser = argparse.ArgumentParser(
        description="Download and partition TensorFlow Speech Commands dataset")
    parser.add_argument(
        '--data_dir',
        type=Path,
        required=True,
        help='Directory where the dataset will be downloaded and partitioned.'
    )
    parser.add_argument(
        '--url',
        type=str,
        default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
        help='URL of the Speech Commands tar.gz archive.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=float,
        default=10.0,
        help='Percentage of files to reserve for validation.'
    )
    parser.add_argument(
        '--testing_percentage',
        type=float,
        default=10.0,
        help='Percentage of files to reserve for testing.'
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    download_dir = data_dir / 'raw'
    extract_dir = data_dir / 'raw'
    output_dir = data_dir / 'partitioned'

    data_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = download_dir / 'speech_commands.tar.gz'
    download_dataset(args.url, archive_path)
    extract_archive(archive_path, extract_dir)

    # The tarball extracts into a folder named 'speech_commands_v0.01'
    raw_dataset_dir = extract_dir / 'speech_commands_v0.01'
    if not raw_dataset_dir.exists():
        # Fallback: check for speech_commands folder
        raw_dataset_dir = extract_dir

    partition_dataset(raw_dataset_dir,
                      output_dir,
                      args.validation_percentage,
                      args.testing_percentage)

if __name__ == '__main__':
    main()
