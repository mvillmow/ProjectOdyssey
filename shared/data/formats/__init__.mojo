"""Data Format Loaders

Provides functions for loading various data formats used in ML Odyssey.

Modules:
    `idx_loader`: IDX file format (MNIST, EMNIST, CIFAR-10 as IDX)
    `cifar_loader`: CIFAR-10 dataset utilities
"""

# IDX Format Loaders
from .idx_loader import (
    read_uint32_be,
    load_idx_labels,
    load_idx_images,
    load_idx_images_rgb,
)

# CIFAR-10 Specific Loaders
from .cifar_loader import (
    normalize_images_rgb,
    load_cifar10_batch,
)
