"""
Data Processing Library

Provides data loading, preprocessing, and augmentation utilities for ML Odyssey.

Modules:
    datasets: Dataset abstractions and common dataset implementations
    loaders: Data loading utilities with batching and shuffling
    transforms: Data transformation and augmentation functions
    samplers: Sampling strategies for data iteration

Example:
    from shared.data import TensorDataset, BatchLoader, Normalize, ToTensor, Compose

    # Create transforms pipeline
    transform = Compose([
        ToTensor(),
        Normalize(mean=0.5, std=0.5),
    ])

    # Create dataset and loader
    dataset = TensorDataset(data, labels)
    loader = BatchLoader(dataset, batch_size=32, shuffle=True)

    # Iterate over batches
    for batch in loader:
        inputs = batch.data
        targets = batch.labels
        # ... training code
"""

# Package version
alias VERSION = "0.1.0"

# ============================================================================
# Exports
# ============================================================================

# Dataset abstractions and implementations
from .datasets import (
    Dataset,  # Base trait
    TensorDataset,  # In-memory tensor dataset
    FileDataset,  # File-based dataset
)

# Data loaders
from .loaders import (
    Batch,  # Batch container
    BaseLoader,  # Base loader functionality
    BatchLoader,  # Main data loader with batching
)

# Sampling strategies
from .samplers import (
    Sampler,  # Base sampler trait
    SequentialSampler,  # Sequential sampling
    RandomSampler,  # Random sampling
    WeightedSampler,  # Weighted sampling
)

# Transforms
from .transforms import (
    Transform,  # Base transform trait
    Compose,  # Compose multiple transforms
    ToTensor,  # Convert to tensor
    Normalize,  # Normalize data
    Reshape,  # Reshape tensor
    Resize,  # Resize images
    CenterCrop,  # Center crop
    RandomCrop,  # Random crop augmentation
    RandomHorizontalFlip,  # Random horizontal flip
    RandomRotation,  # Random rotation augmentation
)

# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Datasets
    "Dataset",
    "TensorDataset",
    "FileDataset",
    # Loaders
    "Batch",
    "BaseLoader",
    "BatchLoader",
    # Samplers
    "Sampler",
    "SequentialSampler",
    "RandomSampler",
    "WeightedSampler",
    # Transforms
    "Transform",
    "Compose",
    "ToTensor",
    "Normalize",
    "Reshape",
    "Resize",
    "CenterCrop",
    "RandomCrop",
    "RandomHorizontalFlip",
    "RandomRotation",
]
