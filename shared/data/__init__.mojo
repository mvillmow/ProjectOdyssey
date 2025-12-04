"""Data Loading Package

Provides data loaders for common ML datasets and utilities for handling various data formats.

Modules:
    `formats`: Low-level data format loaders (IDX, CIFAR, etc.)
    `datasets`: High-level dataset interfaces (CIFAR-10, EMNIST, etc.)

Architecture:
    - `formats` provides low-level file I/O and format parsing
    - `datasets` provides high-level, user-friendly interfaces
    - All data is returned as ExTensor for consistency with core library

Example:
    from shared.data import load_emnist_train, load_emnist_test

    # Load EMNIST
    images, labels = load_emnist_train("/path/to/emnist", split="balanced")

    # Or use the EMNISTDataset class directly
    from shared.data import EMNISTDataset
    dataset = EMNISTDataset("/path/to/emnist", split="balanced", train=True)
    sample_img, sample_label = dataset[0]
"""

# Package version
alias VERSION = "0.1.0"

# ============================================================================
# Format Loaders (Low-Level File I/O)
# ============================================================================

# IDX format utilities
from .formats import (
    read_uint32_be,         # Read big-endian uint32
    load_idx_labels,        # Load IDX label file
    load_idx_images,        # Load IDX grayscale images
    load_idx_images_rgb,    # Load IDX RGB images
    normalize_images,       # Normalize uint8 images to [0, 1] float32
    normalize_images_rgb,   # Normalize RGB images with ImageNet parameters
    one_hot_encode,         # One-hot encode integer labels
    load_cifar10_batch,     # Load single CIFAR-10 batch
)

# ============================================================================
# Dataset Constants and Metadata
# ============================================================================

# Dataset-specific constants and metadata
from .constants import (
    CIFAR10_CLASS_NAMES,           # CIFAR-10 class names (10 classes)
    EMNIST_BALANCED_CLASSES,       # EMNIST Balanced class names (47 classes)
    EMNIST_BYCLASS_CLASSES,        # EMNIST By Class class names (62 classes)
    EMNIST_BYMERGE_CLASSES,        # EMNIST By Merge class names (36 classes)
    EMNIST_DIGITS_CLASSES,         # EMNIST Digits class names (10 classes)
    EMNIST_LETTERS_CLASSES,        # EMNIST Letters class names (52 classes)
    DatasetInfo,                   # Dataset metadata container
)

# ============================================================================
# Dataset Loaders (High-Level Interfaces)
# ============================================================================

# Dataset classes and loaders
from .datasets import (
    Dataset,                # Base dataset interface
    ExTensorDataset,        # In-memory tensor dataset wrapper
    FileDataset,            # File-based lazy-loading dataset
    CIFAR10Dataset,         # CIFAR-10 dataset with train/test splits
)

# EMNIST dataset is defined in _datasets_core.mojo
from ._datasets_core import (
    EMNISTDataset,          # EMNIST dataset with multiple splits
    load_emnist_train,      # Load EMNIST training set
    load_emnist_test,       # Load EMNIST test set
)

# ============================================================================
# Public API
# ============================================================================

# Note: Mojo does not support __all__ for controlling exports.
# All imported symbols are automatically available to package consumers.
#
# High-level usage:
#   from shared.data import load_emnist_train, EMNISTDataset
#   images, labels = load_emnist_train("/path/to/emnist", split="balanced")
#
# Low-level usage:
#   from shared.data import load_idx_images, load_idx_labels, read_uint32_be
#   images = load_idx_images("/path/to/custom-images-idx3-ubyte")

# ============================================================================
# Batch Processing Utilities
# ============================================================================

from .batch_utils import (
    extract_batch,
    extract_batch_pair,
    compute_num_batches,
    get_batch_indices,
)
