#!/usr/bin/env python3
"""
Script to convert WAV audio files into fixed-shape FFT feature arrays and save them as .npy files.

Usage:
    python transform_to_fft.py \
        --input_dir /path/to/raw_partitioned \
        --output_dir /path/to/fft_features \
        [--sr 16000] \
        [--max_length 16000] \
        [--verbose]

This script pads or trims each audio to `max_length` samples, computes the real FFT,
separates real and imaginary parts, and ensures a consistent feature length across all files.
"""
import argparse
from pathlib import Path
import numpy as np
import librosa

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert WAV files to fixed-shape FFT .npy files"
    )
    parser.add_argument(
        '--input_dir', '-i',
        type=Path,
        required=True,
        help='Path to input directory containing partitioned WAV files'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=Path,
        required=True,
        help='Path to output directory where FFT arrays will be saved'
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=16000,
        help='Sampling rate for audio loading (default: 16000)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=None,
        help='Fixed length (in samples) to pad/trim audio (default: sr)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=False,
        help='Enable verbose output'
    )
    return parser.parse_args()


def process_file(
    wav_path: Path,
    fft_path: Path,
    sr: int,
    max_length: int,
    verbose: bool
):
    # Load and fix audio length
    y, _ = librosa.load(str(wav_path), sr=sr)
    length = max_length or sr
    y = librosa.util.fix_length(y, size=length)

    # Compute real FFT
    Y = np.fft.rfft(y)
    # Separate real and imaginary parts
    real = Y.real.astype(np.float32)
    imag = Y.imag.astype(np.float32)
    # Stack to shape (2, freq_bins)
    features = np.stack([real, imag], axis=0)

    # Save as .npy
    out_file = fft_path.with_suffix('.npy')
    np.save(str(out_file), features)
    if verbose:
        print(f"Saved FFT features {out_file} shape={features.shape}")


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    sr = args.sr
    max_length = args.max_length or sr
    verbose = args.verbose

    # Process splits if they exist
    splits = ['training', 'validation', 'testing']
    for split in splits:
        in_split = input_dir / split
        out_split = output_dir / split
        if not in_split.exists():
            continue
        for label_dir in in_split.iterdir():
            if not label_dir.is_dir():
                continue
            out_label = out_split / label_dir.name
            out_label.mkdir(parents=True, exist_ok=True)
            for wav_file in label_dir.glob('*.wav'):
                process_file(
                    wav_file,
                    out_label / wav_file.name,
                    sr,
                    max_length,
                    verbose
                )
    print("FFT feature extraction complete.")

if __name__ == '__main__':
    main()
