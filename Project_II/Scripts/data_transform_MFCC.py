#!/usr/bin/env python3
"""
Script to convert WAV audio files into MFCC feature arrays of fixed shape and save them as .npy files.

Usage:
    python transform_to_mfcc.py \
        --input_dir /path/to/raw_partitioned \
        --output_dir /path/to/mfccs \
        [--sr 16000] \
        [--max_length 16000] \
        [--n_mfcc 13] \
        [--n_fft 512] \
        [--hop_length 256] \
        [--win_length 512] \
        [--fmin 0] \
        [--fmax None] \
        [--verbose]

This script first pads or truncates each audio to `max_length` samples,
ensuring consistent MFCC time dimension across all files.
"""
import argparse
from pathlib import Path
import numpy as np
import librosa

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert WAV files to fixed-shape MFCC .npy files"
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
        help='Path to output directory where MFCC arrays will be saved'
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
        '--n_mfcc',
        type=int,
        default=13,
        help='Number of MFCCs to extract (default: 13)'
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
        default=512,
        help='Window length (default: 512)'
    )
    parser.add_argument(
        '--fmin',
        type=float,
        default=0.0,
        help='Minimum frequency for mel filters (default: 0.0)'
    )
    parser.add_argument(
        '--fmax',
        type=float,
        default=None,
        help='Maximum frequency for mel filters (default: sr/2)'
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
    mfcc_path: Path,
    sr: int,
    max_length: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    fmin: float,
    fmax: float,
    verbose: bool
):
    # Load audio and fix length
    y, _ = librosa.load(str(wav_path), sr=sr)
    length = max_length or sr
    y = librosa.util.fix_length(y, size=length)
    # Compute mel-spectrogram
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax if fmax is not None else sr/2
    )
    # Compute MFCCs
    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(S),
        n_mfcc=n_mfcc
    )
    # Save as .npy
    out_file = mfcc_path.with_suffix('.npy')
    np.save(str(out_file), mfcc.astype(np.float32))
    if verbose:
        print(f"Saved MFCC {out_file} shape={mfcc.shape}")


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    sr = args.sr
    max_length = args.max_length or sr
    n_mfcc = args.n_mfcc
    n_fft = args.n_fft
    hop_length = args.hop_length
    win_length = args.win_length
    fmin = args.fmin
    fmax = args.fmax
    verbose = args.verbose

    for split in ['training', 'validation', 'testing']:
        in_split = input_dir / split
        out_split = output_dir / split
        for label_dir in in_split.iterdir():
            if not label_dir.is_dir():
                continue
            out_label = out_split / label_dir.name
            out_label.mkdir(parents=True, exist_ok=True)
            for wav_file in label_dir.glob('*.wav'):
                mfcc_file = out_label / wav_file.name
                process_file(
                    wav_file,
                    mfcc_file,
                    sr,
                    max_length,
                    n_mfcc,
                    n_fft,
                    hop_length,
                    win_length,
                    fmin,
                    fmax,
                    verbose
                )
    print("MFCC extraction with fixed shape complete.")

if __name__ == '__main__':
    main()
