#!/usr/bin/env python3
"""
Script to convert WAV audio files into fixed-shape log-magnitude spectrogram arrays and save them as .npy files.

Usage:
    python transform_to_spectrogram.py \
        --input_dir /path/to/raw_partitioned \
        --output_dir /path/to/spectrograms \
        [--sr 16000] \
        [--max_length 16000] \
        [--n_fft 512] \
        [--hop_length 256] \
        [--win_length 512] \
        [--window hann] \
        [--verbose]

This script pads or trims each audio to `max_length` samples, computes the STFT,
converts to log-magnitude spectrograms, and ensures a consistent time dimension for all files.
"""
import argparse
from pathlib import Path
import numpy as np
import librosa


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert WAV files to fixed-shape spectrogram .npy files"
    )
    parser.add_argument(
        '--input_dir', '-i',
        type=Path,
        required=True,
        help='Path to input directory containing partitioned WAV files (training/validation/testing)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=Path,
        required=True,
        help='Path to output directory where spectrogram arrays will be saved'
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
        '--n_fft',
        type=int,
        default=512,
        help='FFT window size (default: 512)'
    )
    parser.add_argument(
        '--hop_length',
        type=int,
        default=256,
        help='Hop length (default: 256)'
    )
    parser.add_argument(
        '--win_length',
        type=int,
        default=None,
        help='Window length (default: n_fft)'
    )
    parser.add_argument(
        '--window',
        type=str,
        default='hann',
        help='Window function for STFT (default: hann)'
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
    spec_path: Path,
    sr: int,
    max_length: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
    verbose: bool
):
    # Load and fix audio length
    y, _ = librosa.load(str(wav_path), sr=sr)
    length = max_length or sr
    y = librosa.util.fix_length(y, size=length)
    # Compute STFT and magnitude
    S = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window
    )
    S_mag = np.abs(S)
    # Convert to log scale (dB)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
    # Save as .npy
    out_file = spec_path.with_suffix('.npy')
    np.save(str(out_file), S_db.astype(np.float32))
    if verbose:
        print(f"Saved spectrogram {out_file} shape={S_db.shape}")


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    sr = args.sr
    max_length = args.max_length or sr
    n_fft = args.n_fft
    hop_length = args.hop_length
    win_length = args.win_length or n_fft
    window = args.window
    verbose = args.verbose

    # Process by split
    for split in ['training', 'validation', 'testing']:
        in_split = input_dir / split
        out_split = output_dir / split
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
                    n_fft,
                    hop_length,
                    win_length,
                    window,
                    verbose
                )
    print("Spectrogram extraction complete.")


if __name__ == '__main__':
    main()
