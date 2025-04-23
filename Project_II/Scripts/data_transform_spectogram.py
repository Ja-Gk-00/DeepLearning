#!/usr/bin/env python3
"""
Script to convert WAV audio files into spectrograms and save them as .npy files.

Usage:
    python transform_to_spectrogram.py \
        --input_dir /path/to/raw_partitioned \
        --output_dir /path/to/spectrograms \
        [--sr 16000] \
        [--n_fft 512] \
        [--hop_length 256] \
        [--win_length 512]
"""
import argparse
import os
from pathlib import Path
import numpy as np
import librosa


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert WAV files to spectrogram .npy files"
    )
    parser.add_argument(
        '--input_dir', '-i',
        type=Path,
        required=True,
        help='Path to input directory containing partitioned WAV files (e.g., training/validation/testing)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=Path,
        required=True,
        help='Path to output directory where spectrograms will be saved'
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=16000,
        help='Sampling rate for audio loading (default: 16000)'
    )
    parser.add_argument(
        '--n_fft',
        type=int,
        default=512,
        help='FFT window size for STFT (default: 512)'
    )
    parser.add_argument(
        '--hop_length',
        type=int,
        default=256,
        help='Hop length (stride) for STFT (default: 256)'
    )
    parser.add_argument(
        '--win_length',
        type=int,
        default=512,
        help='Window length for STFT (default: 512)'
    )
    return parser.parse_args()


def process_file(wav_path: Path, spec_path: Path, sr: int, n_fft: int, hop_length: int, win_length: int):
    y, _ = librosa.load(wav_path, sr=sr)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    np.save(str(spec_path.with_suffix('.npy')), S_db)


def main():
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    sr = args.sr
    n_fft = args.n_fft
    hop_length = args.hop_length
    win_length = args.win_length

    # Walk through splits and labels
    for split in ['training', 'validation', 'testing']:
        split_in = input_dir / split
        split_out = output_dir / split
        for label_dir in split_in.iterdir():
            if label_dir.is_dir():
                label_out = split_out / label_dir.name
                label_out.mkdir(parents=True, exist_ok=True)
                # Process each wav
                for wav_file in label_dir.glob('*.wav'):
                    spec_file = label_out / wav_file.name
                    print(f"Processing {wav_file} -> {spec_file.with_suffix('.npy')}")
                    try:
                        process_file(wav_file, spec_file, sr, n_fft, hop_length, win_length)
                    except Exception as e:
                        print(f"Error processing {wav_file}: {e}")

    print("Spectrogram conversion complete.")


if __name__ == '__main__':
    main()
