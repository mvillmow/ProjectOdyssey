#!/usr/bin/env python3
"""
Download MNIST dataset for ML Odyssey.

Justification: Python is used for HTTP downloads and gzip extraction (Mojo v0.26.1
limitation: subprocess API lacks proper exit code and output capture).
See: docs/adr/ADR-001-language-selection-tooling.md

Downloads and extracts MNIST dataset in IDX format.

Usage:
    python scripts/download_mnist.py [output_dir]

Default output: datasets/mnist
"""

import gzip
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# MNIST dataset URL
MNIST_BASE_URL = "http://yann.lecun.com/exdb/mnist"

# Files to download: (filename, description)
MNIST_FILES = [
    ("train-images-idx3-ubyte.gz", "training images"),
    ("train-labels-idx1-ubyte.gz", "training labels"),
    ("t10k-images-idx3-ubyte.gz", "test images"),
    ("t10k-labels-idx1-ubyte.gz", "test labels"),
]

# User-Agent header to avoid bot blocking
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # Exponential backoff: 1s, 2s, 4s


def download_with_retry(url: str, output_path: Path, max_retries: int = MAX_RETRIES) -> None:
    """
    Download file with User-Agent header and retry logic.

    Uses exponential backoff for retries to handle transient failures.

    Args:
        url: URL to download from
        output_path: Path to save downloaded file
        max_retries: Maximum number of retry attempts

    Raises:
        RuntimeError: If download fails after all retries
    """
    last_error = None

    for attempt in range(max_retries):
        if attempt > 0:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            print(f"  Retry {attempt}/{max_retries - 1} after {delay}s delay...")
            time.sleep(delay)

        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                block_size = 8192

                with open(output_path, "wb") as f:
                    while True:
                        block = response.read(block_size)
                        if not block:
                            break
                        f.write(block)
                        downloaded += len(block)

                        # Progress bar
                        if total_size > 0:
                            percent = min(100, downloaded * 100 / total_size)
                            bar_length = 50
                            filled = int(bar_length * downloaded / total_size)
                            bar = "=" * filled + "-" * (bar_length - filled)
                            print(
                                f"\rDownloading: [{bar}] {percent:.1f}%",
                                end="",
                                flush=True,
                            )

                print()  # New line after progress bar
                return

        except HTTPError as e:
            last_error = f"HTTP {e.code}: {e.reason}"
            print(f"\n  Download failed: {last_error}")
        except URLError as e:
            last_error = f"URL Error: {e.reason}"
            print(f"\n  Download failed: {last_error}")
        except Exception as e:
            last_error = str(e)
            print(f"\n  Download failed: {last_error}")

    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts. Last error: {last_error}")


def decompress_gz(gz_path: Path, output_path: Path) -> None:
    """
    Decompress gzip file.

    Args:
        gz_path: Path to .gz file
        output_path: Path to save decompressed file
    """
    with gzip.open(gz_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            f_out.write(f_in.read())


def download_mnist(output_dir: str):
    """Download and extract MNIST dataset.

    Args:
        output_dir: Directory to save dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # MNIST file mappings: (source_filename, output_filename)
    file_mappings = [
        ("train-images-idx3-ubyte.gz", "train_images.idx"),
        ("train-labels-idx1-ubyte.gz", "train_labels.idx"),
        ("t10k-images-idx3-ubyte.gz", "test_images.idx"),
        ("t10k-labels-idx1-ubyte.gz", "test_labels.idx"),
    ]

    for gz_filename, output_filename in file_mappings:
        gz_path = output_path / gz_filename
        output_file_path = output_path / output_filename

        # Download if not exists
        if not output_file_path.exists():
            url = f"{MNIST_BASE_URL}/{gz_filename}"
            print(f"Downloading MNIST {output_filename.split('_')[0]} data...")
            print(f"  From: {url}")

            download_with_retry(url, gz_path)

            # Decompress
            print(f"Decompressing {gz_filename}...")
            decompress_gz(gz_path, output_file_path)

            # Clean up gzip file
            gz_path.unlink()
            print(f"✓ {output_filename} ready")
        else:
            print(f"✓ {output_filename} already exists")

    print(f"\n✓ MNIST dataset ready at: {output_path}")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "datasets/mnist"
    download_mnist(output_dir)
