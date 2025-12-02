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
    from shared.data import load_cifar10_train, load_cifar10_test, load_emnist_train, load_emnist_test

    # Load CIFAR-10
    images, labels = load_cifar10_train("/path/to/cifar10")

    # Load EMNIST with normalization
    images_raw, labels = load_emnist_train("/path/to/emnist")
    images_norm = normalize_images(images_raw)
    labels_onehot = one_hot_encode(labels, 62)  # 62 EMNIST classes
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
    normalize_images_rgb,   # Normalize RGB images
    load_cifar10_batch,     # Load single CIFAR-10 batch
)

# ============================================================================
# Dataset Loaders (High-Level Interfaces)
# ============================================================================

# CIFAR-10 dataset
from .datasets import (
    load_cifar10_train,     # Load CIFAR-10 training set
    load_cifar10_test,      # Load CIFAR-10 test set
    load_emnist_train,      # Load EMNIST training set
    load_emnist_test,       # Load EMNIST test set
    normalize_images,       # Normalize grayscale images to [0, 1]
    one_hot_encode,         # Convert labels to one-hot encoding
)

# ============================================================================
# Public API
# ============================================================================

# Note: Mojo does not support __all__ for controlling exports.
# All imported symbols are automatically available to package consumers.
#
# High-level usage:
#   from shared.data import load_cifar10_train, load_emnist_test
#   images, labels = load_cifar10_train("/path/to/data")
#
# Low-level usage:
#   from shared.data import load_idx_images_rgb, read_uint32_be
#   images = load_idx_images_rgb("/path/to/custom.idx")
