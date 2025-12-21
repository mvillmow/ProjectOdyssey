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
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# CIFAR-10 dataset URL
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

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
        download_with_retry(CIFAR10_URL, tar_path)
        print("Download complete!")
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
