#!/usr/bin/env python3
"""
Script to convert WAV audio files into FFT-based frequency distributions and save them as .npz files.

Usage:
    python transform_to_fft.py \
        --input_dir /path/to/raw_partitioned \
        --output_dir /path/to/fft_outputs \
        [--sr 16000] \
        [--n_fft None] \
        [--verbose]

Each output .npz contains two arrays:
  - freq: frequency bins (Hz)
  - magnitude: corresponding magnitude of the FFT
"""
import argparse
from pathlib import Path
import numpy as np
import librosa

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert WAV files to FFT-based frequency distributions (.npz)"
    )
    parser.add_argument(
        '--input_dir', '-i',
        type=Path,
        required=True,
        help='Directory with partitioned WAV files (training/validation/testing)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=Path,
        required=True,
        help='Directory where FFT outputs will be saved'
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=16000,
        help='Sampling rate for loading audio (default: 16000)'
    )
    parser.add_argument(
        '--n_fft',
        type=int,
        default=None,
        help='FFT size (default: signal length)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=False,
        help='Enable verbose output'
    )
    return parser.parse_args()

def process_file(wav_path: Path, out_path: Path, sr: int, n_fft: int):
    # Load audio
    y, _ = librosa.load(wav_path, sr=sr)
    # Determine FFT length
    n = n_fft if n_fft is not None else len(y)
    # Compute real FFT
    fft_vals = np.fft.rfft(y, n=n)
    magnitude = np.abs(fft_vals)
    # Frequency bins
    freq = np.fft.rfftfreq(n, d=1.0/sr)
    # Save both arrays in a compressed npz
    np.savez(out_path.with_suffix('.npz'), freq=freq, magnitude=magnitude)


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    sr = args.sr
    n_fft = args.n_fft
    verbose = args.verbose

    # Walk through splits and labels
    for split in ['training', 'validation', 'testing']:
        split_in = input_dir / split
        split_out = output_dir / split
        for label_dir in split_in.iterdir():
            if label_dir.is_dir():
                label_out = split_out / label_dir.name
                label_out.mkdir(parents=True, exist_ok=True)
                # Process each wav file
                for wav_file in label_dir.glob('*.wav'):
                    out_file = label_out / wav_file.name
                    if verbose:
                        print(f"FFT processing {wav_file} -> {out_file.with_suffix('.npz')}")
                    try:
                        process_file(wav_file, out_file, sr, n_fft)
                    except Exception as e:
                        print(f"Error processing {wav_file}: {e}")

    print("FFT extraction complete.")

if __name__ == '__main__':
    main()
