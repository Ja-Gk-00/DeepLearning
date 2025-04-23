#!/usr/bin/env python3
"""
Script to extract multiple moving-window features from WAV audio files and save them as .npz files.

Features computed (each over sliding windows):
  1. Amplitude mean (mean absolute value)
  2. Amplitude standard deviation
  3. Zero-crossing rate
  4. Root-mean-square (RMS) energy
  5. Spectral centroid
  6. Spectral bandwidth
  7. Spectral rolloff

Usage:
    python transform_moving_functions.py \
        --input_dir /path/to/data/partitioned \
        --output_dir /path/to/moving_features \
        [--sr 16000] \
        [--num_steps 1000] \
        [--window_size None] \
        [--verbose]

Arguments:
  --input_dir    Directory with partitioned WAV files (training/validation/testing)
  --output_dir   Directory where feature files will be saved
  --sr           Sampling rate for loading (default: 16000)
  --num_steps    Number of windowed steps/points per file (default: 1000)
  --window_size  Window size in samples (default: len(y)//num_steps)
  --verbose, -v  Show progress messages
"""
import argparse
from pathlib import Path
import numpy as np
import librosa


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract moving-window features from WAV files"
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
        '--num_steps',
        type=int,
        default=1000,
        help='Number of windowed steps/points per file (default: 1000)'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=None,
        help='Window size in samples (default: len(y)//num_steps)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=False,
        help='Enable verbose output'
    )
    return parser.parse_args()


def compute_amplitude_mean(y, window_size, hop_length, num_frames):
    return np.array([
        np.mean(np.abs(y[i*hop_length : i*hop_length + window_size]))
        for i in range(num_frames)
    ], dtype=np.float32)


def compute_amplitude_std(y, window_size, hop_length, num_frames):
    return np.array([
        np.std(y[i*hop_length : i*hop_length + window_size])
        for i in range(num_frames)
    ], dtype=np.float32)


def compute_zero_crossing_rate(y, window_size, hop_length):
    return librosa.feature.zero_crossing_rate(
        y,
        frame_length=window_size,
        hop_length=hop_length,
        center=False
    )[0]


def compute_rms(y, window_size, hop_length):
    return librosa.feature.rms(
        y=y,
        frame_length=window_size,
        hop_length=hop_length,
        center=False
    )[0]


def compute_spectral_centroid(y, sr, window_size, hop_length):
    return librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=window_size,
        hop_length=hop_length,
        center=False
    )[0]


def compute_spectral_bandwidth(y, sr, window_size, hop_length):
    return librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        n_fft=window_size,
        hop_length=hop_length,
        center=False
    )[0]


def compute_spectral_rolloff(y, sr, window_size, hop_length, roll_percent=0.85):
    return librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        n_fft=window_size,
        hop_length=hop_length,
        roll_percent=roll_percent,
        center=False
    )[0]


def compute_features(y, sr, num_steps, window_size=None):
    """
    Compute all seven features over sliding windows. Returns matrix shape (7, num_frames).
    """
    L = len(y)
    if window_size is None:
        window_size = max(1, L // num_steps)
    hop_length = window_size
    num_frames = int(np.floor((L - window_size) / hop_length)) + 1

    f1 = compute_amplitude_mean(y, window_size, hop_length, num_frames)
    f2 = compute_amplitude_std(y, window_size, hop_length, num_frames)
    f3 = compute_zero_crossing_rate(y, window_size, hop_length)
    f4 = compute_rms(y, window_size, hop_length)
    f5 = compute_spectral_centroid(y, sr, window_size, hop_length)
    f6 = compute_spectral_bandwidth(y, sr, window_size, hop_length)
    f7 = compute_spectral_rolloff(y, sr, window_size, hop_length)

    return np.vstack([f1, f2, f3, f4, f5, f6, f7])


def process_file(wav_path, out_path, sr, num_steps, window_size, verbose):
    y, _ = librosa.load(wav_path, sr=sr)
    features = compute_features(y, sr, num_steps, window_size)
    np.savez(out_path.with_suffix('.npz'), features=features)
    if verbose:
        print(f"Saved {out_path.with_suffix('.npz')} with shape {features.shape}")


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    sr = args.sr
    num_steps = args.num_steps
    window_size = args.window_size
    verbose = args.verbose

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
                    process_file(wav_file, out_file, sr, num_steps, window_size, verbose)

    print("Moving-window feature extraction complete.")

if __name__ == '__main__':
    main()
