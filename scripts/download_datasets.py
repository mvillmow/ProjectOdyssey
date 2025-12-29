#!/usr/bin/env python3
"""
Unified dataset downloader for ML Odyssey.

Justification: Python is used for HTTP downloads and archive extraction (Mojo v0.26.1
limitation: subprocess API lacks proper exit code and output capture).
See: docs/adr/ADR-001-language-selection-tooling.md

Provides a single entry point to download all standard datasets for ML Odyssey.

Usage:
    python scripts/download_datasets.py --list
    python scripts/download_datasets.py mnist
    python scripts/download_datasets.py fashion_mnist
    python scripts/download_datasets.py cifar10
    python scripts/download_datasets.py cifar100
    python scripts/download_datasets.py all

Default output: datasets/<dataset_name>
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import Optional

# Available datasets and their download scripts
DATASETS = {
    "mnist": {
        "script": "scripts/download_mnist.py",
        "description": "MNIST - 60k training, 10k test 28x28 grayscale digits",
        "default_dir": "datasets/mnist",
    },
    "fashion_mnist": {
        "script": "scripts/download_fashion_mnist.py",
        "description": "Fashion-MNIST - 60k training, 10k test 28x28 grayscale clothing",
        "default_dir": "datasets/fashion_mnist",
    },
    "cifar10": {
        "script": "scripts/download_cifar10.py",
        "description": "CIFAR-10 - 50k training, 10k test 32x32 RGB images (10 classes)",
        "default_dir": "datasets/cifar10",
    },
    "cifar100": {
        "script": "scripts/download_cifar100.py",
        "description": "CIFAR-100 - 50k training, 10k test 32x32 RGB images (100 classes)",
        "default_dir": "datasets/cifar100",
    },
}


def list_datasets() -> None:
    """List all available datasets."""
    print("\nAvailable datasets:\n")
    for name, info in DATASETS.items():
        print(f"  {name:20} {info['description']}")
    print("\nUsage:")
    print("  python scripts/download_datasets.py <dataset_name> [output_dir]")
    print("  python scripts/download_datasets.py all                      # Download all datasets")
    print()


def download_dataset(name: str, output_dir: Optional[str] = None) -> int:
    """
    Download a single dataset.

    Args:
        name: Dataset name
        output_dir: Optional output directory

    Returns:
        0 if successful, 1 if failed
    """
    if name not in DATASETS:
        print(f"Error: Unknown dataset '{name}'")
        print("Use --list to see available datasets")
        return 1

    dataset_info = DATASETS[name]
    script_path = dataset_info["script"]
    default_dir = dataset_info["default_dir"]

    # Determine output directory
    if output_dir is None:
        output_dir = default_dir

    # Check if script exists
    if not Path(script_path).exists():
        print(f"Error: Download script not found: {script_path}")
        return 1

    # Run download script
    print(f"\n{'=' * 60}")
    print(f"Downloading {name}...")
    print(f"{'=' * 60}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_path, output_dir],
            check=False,
        )
        return result.returncode
    except Exception as e:
        print(f"Error: Failed to run download script: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download datasets for ML Odyssey",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_datasets.py --list
  python scripts/download_datasets.py mnist
  python scripts/download_datasets.py cifar10 /custom/path
  python scripts/download_datasets.py all
        """,
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset name (mnist, fashion_mnist, cifar10, cifar100, all) or --list",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )

    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Output directory (optional)",
    )

    args = parser.parse_args()

    # Handle --list flag
    if args.list:
        list_datasets()
        return 0

    # Require dataset name
    if args.dataset is None:
        parser.print_help()
        return 1

    # Handle special case: list (for backwards compatibility)
    if args.dataset == "--list":
        list_datasets()
        return 0

    # Handle 'all' - download all datasets
    if args.dataset == "all":
        failed = []
        for dataset_name in DATASETS.keys():
            ret = download_dataset(dataset_name, None)
            if ret != 0:
                failed.append(dataset_name)

        if failed:
            print(f"\n❌ Failed to download: {', '.join(failed)}")
            return 1
        else:
            print("\n✓ All datasets downloaded successfully!")
            return 0

    # Download single dataset
    if args.dataset not in DATASETS:
        print(f"Error: Unknown dataset '{args.dataset}'")
        list_datasets()
        return 1

    ret = download_dataset(args.dataset, args.output_dir)
    return ret


if __name__ == "__main__":
    sys.exit(main())
