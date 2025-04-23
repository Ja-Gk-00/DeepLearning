#!/usr/bin/env python3
"""
Script to convert WAV audio files into time-frequency scattering transform coefficients and save them as .npz files.

Requires:
    pip install kymatio librosa numpy

Usage:
    python transform_to_scattering.py \
        --input_dir /path/to/data/partitioned \
        --output_dir /path/to/scattering_features \
        [--sr 16000] \
        [--J 6] \
        [--Q 8] \
        [--max_length 16000] \
        [--verbose]

Arguments:
  --input_dir    Directory with partitioned WAV files (training/validation/testing)
  --output_dir   Directory where scattering feature files will be saved
  --sr           Sampling rate for audio loading (default: 16000)
  --J            Scale of scattering (log2 of maximum scale, controls temporal support)
  --Q            Number of wavelets per octave (quality factor)
  --max_length   Length in samples to pad/trim audio (default: sr, i.e., 1 second)
  --verbose, -v  Show progress messages
"""
import argparse
from pathlib import Path
import numpy as np
import librosa
from kymatio.torch import Scattering1D
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract scattering transform features from WAV files"
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
        help='Path to output directory where feature arrays will be saved'
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=16000,
        help='Sampling rate for audio loading (default: 16000)'
    )
    parser.add_argument(
        '--J',
        type=int,
        default=6,
        help='Log2 scale of scattering (default: 6)'
    )
    parser.add_argument(
        '--Q',
        type=int,
        default=8,
        help='Quality factor (wavelets per octave, default: 8)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=None,
        help='Fixed length in samples to pad/trim audio (default: sr samples)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=False,
        help='Enable verbose output'
    )
    return parser.parse_args()

def process_file(wav_path: Path, out_path: Path, scattering, sr, max_length, verbose):
    # Load audio
    y, _ = librosa.load(str(wav_path), sr=sr)
    # Pad or trim to fixed length
    if max_length is not None:
        y = librosa.util.fix_length(y, size=max_length)
    # Convert to torch tensor
    x = torch.from_numpy(y.astype(np.float32))
    # Compute scattering coefficients (C, T)
    Sx = scattering(x)
    # Convert to numpy
    Sx_np = Sx.numpy()
    # Save scattering as compressed npz
    np.savez(str(out_path.with_suffix('.npz')), scattering=Sx_np)
    if verbose:
        print(f"Saved {out_path.with_suffix('.npz')} with shape {Sx_np.shape}")


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    sr = args.sr
    J = args.J
    Q = args.Q
    max_length = args.max_length or sr
    verbose = args.verbose

    # Create scattering transform object
    # We pass the maximum length as 'shape'
    scattering = Scattering1D(J=J, shape=(max_length,), Q=Q)

    for split in ['training', 'validation', 'testing']:
        split_in = input_dir / split
        split_out = output_dir / split
        for label_dir in split_in.iterdir():
            if label_dir.is_dir():
                label_out = split_out / label_dir.name
                label_out.mkdir(parents=True, exist_ok=True)
                for wav_file in label_dir.glob('*.wav'):
                    out_file = label_out / wav_file.name
                    if verbose:
                        print(f"Processing {wav_file}")
                    try:
                        process_file(wav_file, out_file, scattering, sr, max_length, verbose)
                    except Exception as e:
                        print(f"Error processing {wav_file}: {e}")
    print("Scattering transform extraction complete.")

if __name__ == '__main__':
    main()
