"""
Data Processing Library

Provides data loading, preprocessing, and augmentation utilities for ML Odyssey.

Modules:
    datasets: Dataset abstractions and common dataset implementations
    loaders: Data loading utilities with batching and shuffling
    transforms: Data transformation and augmentation functions
    samplers: Sampling strategies for data iteration

Example:
    from shared.data import ExTensorDataset, BatchLoader, Normalize, ToExTensor, Compose

    # Create transforms pipeline
    transform = Compose([
        ToExTensor(),
        Normalize(mean=0.5, std=0.5),
    ])

    # Create dataset and loader
    dataset = ExTensorDataset(data, labels)
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
    ExTensorDataset,  # In-memory tensor dataset
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
    ToExTensor,  # Convert to tensor
    Normalize,  # Normalize data
    Reshape,  # Reshape tensor
    Resize,  # Resize images
    CenterCrop,  # Center crop
    RandomCrop,  # Random crop augmentation
    RandomHorizontalFlip,  # Random horizontal flip
    RandomRotation,  # Random rotation augmentation
)

# Batch utilities
from .batch_utils import (
    extract_batch,  # Extract mini-batch from dataset
    extract_batch_pair,  # Extract batch of data and labels
    compute_num_batches,  # Compute number of batches needed
    get_batch_indices,  # Get start/end indices for a batch
)

# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Datasets
    "Dataset",
    "ExTensorDataset",
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
    "ToExTensor",
    "Normalize",
    "Reshape",
    "Resize",
    "CenterCrop",
    "RandomCrop",
    "RandomHorizontalFlip",
    "RandomRotation",
    # Batch utilities
    "extract_batch",
    "extract_batch_pair",
    "compute_num_batches",
    "get_batch_indices",
]
