"""
Data Processing Library

Provides data loading, preprocessing, and augmentation utilities for ML Odyssey.

Modules:
    `datasets`: Dataset abstractions and common dataset implementations.
    `loaders`: Data loading utilities with batching and shuffling.
    `transforms`: Data transformation and augmentation functions.
    `samplers`: Sampling strategies for data iteration.

Example:.    from shared.data import ExTensorDataset, BatchLoader, Normalize, ToExTensor, Compose

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

FIXME: Placeholder import tests in tests/shared/test_imports.mojo require:
- test_data_imports (line 160+)
- test_data_datasets_imports (line 170+)
- test_data_loaders_imports (line 180+)
- test_data_transforms_imports (line 195+)
All tests marked as "(placeholder)" and require uncommented imports as Issue #49 progresses.
See Issue #49 for details
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
    Pipeline,  # Type alias for Compose
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

# Note: Mojo does not support __all__ for controlling exports.
# All imported symbols are automatically available to package consumers.
