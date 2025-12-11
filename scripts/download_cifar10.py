#!/usr/bin/env python3
"""
Download CIFAR-10 dataset for ML Odyssey CI and testing.

This script downloads and extracts the CIFAR-10 dataset.
For full IDX conversion, use the version in examples/alexnet-cifar10/.

Usage:
    python scripts/download_cifar10.py [output_dir]

Default output: datasets/cifar10
"""

import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

# CIFAR-10 dataset URL
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def download_progress(block_num: int, block_size: int, total_size: int):
    """Display download progress."""
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 / total_size)
    bar_length = 50
    filled = int(bar_length * downloaded / total_size)
    bar = "=" * filled + "-" * (bar_length - filled)
    print(f"\rDownloading: [{bar}] {percent:.1f}%", end="", flush=True)


def download_cifar10(output_dir: str):
    """Download and extract CIFAR-10 dataset.

    Args:
        output_dir: Directory to save dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tar_path = output_path / "cifar-10-python.tar.gz"
    extract_dir = output_path / "cifar-10-batches-py"

    # Download if not exists
    if not tar_path.exists():
        print(f"Downloading CIFAR-10 from {CIFAR10_URL}...")
        urlretrieve(CIFAR10_URL, tar_path, reporthook=download_progress)
        print("\nDownload complete!")
    else:
        print(f"CIFAR-10 archive already exists: {tar_path}")

    # Extract if not exists
    if not extract_dir.exists():
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(output_path)
        print("Extraction complete!")
    else:
        print(f"CIFAR-10 already extracted: {extract_dir}")

    print(f"\n✓ CIFAR-10 dataset ready at: {output_path}")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "datasets/cifar10"
    download_cifar10(output_dir)
