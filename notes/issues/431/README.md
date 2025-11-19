# Issue #431: [Package] Data Utils - Integration and Packaging

## Objective

Package the data utilities module with clean public API exports, comprehensive documentation, and usage examples demonstrating dataset interfaces, data loaders, and integration with augmentation pipelines.

## Deliverables

### Package Structure

```
shared/data/
├── __init__.mojo                   # Public API exports
├── datasets.mojo                   # Dataset interfaces and implementations
├── loaders.mojo                    # DataLoader with batching/shuffling
├── samplers.mojo                   # Sampling strategies
├── transforms.mojo                 # Image augmentations
├── text_transforms.mojo            # Text augmentations
├── generic_transforms.mojo         # Generic transforms
└── README.md                       # Module documentation
```

### Public API Exports

**File**: `/home/user/ml-odyssey/shared/data/__init__.mojo`

```mojo
"""Data utilities module.

Provides dataset interfaces, data loaders, samplers, and augmentation
transforms for efficient data preparation in machine learning pipelines.

Components:
    - Datasets: Base interfaces and implementations
    - Loaders: Batching and shuffling utilities
    - Samplers: Index sampling strategies
    - Transforms: Data augmentation (image, text, generic)

Quick Start:
    >>> from shared.data import TensorDataset, DataLoader
    >>>
    >>> # Create dataset
    >>> var dataset = TensorDataset(data, labels)
    >>>
    >>> # Create loader with batching
    >>> var loader = DataLoader(dataset, batch_size=32, shuffle=True)
    >>>
    >>> # Iterate over batches
    >>> for batch in loader:
    ...     # Train on batch
    ...     pass
"""

# Dataset interfaces
from .datasets import (
    Dataset,           # Base dataset trait
    TensorDataset,     # In-memory dataset
    FileDataset,       # File-based dataset (placeholder)
)

# Data loaders and samplers
from .loaders import DataLoader, DataLoaderIterator
from .samplers import (
    Sampler,
    SequentialSampler,
    RandomSampler,
)

# Image augmentations
from .transforms import (
    Transform,
    ToTensor,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    CenterCrop,
    RandomCrop,
    RandomRotation,
    RandomErasing,
    Compose,
    Pipeline,
)

# Text augmentations
from .text_transforms import (
    TextTransform,
    RandomSwap,
    RandomDeletion,
    RandomInsertion,
    RandomSynonymReplacement,
    TextCompose,
    TextPipeline,
    split_words,
    join_words,
)

# Generic transforms
from .generic_transforms import (
    IdentityTransform,
    LambdaTransform,
    ConditionalTransform,
    ClampTransform,
    DebugTransform,
    ToFloat32,
    ToInt32,
    SequentialTransform,
    BatchTransform,
)
```

### Module Documentation

**File**: `/home/user/ml-odyssey/shared/data/README.md`

```markdown
# Data Utilities Module

Comprehensive data handling utilities for machine learning pipelines.

## Overview

The data utilities module provides:

1. **Dataset Interfaces**: Consistent API for data access
2. **Data Loaders**: Efficient batching and shuffling
3. **Augmentation Transforms**: Image, text, and generic transformations
4. **Sampling Strategies**: Flexible index sampling

## Components

### 1. Datasets

#### TensorDataset (In-Memory)

```mojo
from shared.data import TensorDataset

# Create dataset from tensors
var data = Tensor(...)      # Shape: [N, features...]
var labels = Tensor(...)    # Shape: [N]

var dataset = TensorDataset(data, labels)

# Access samples
print(len(dataset))         # Number of samples
var sample = dataset[0]     # Get first sample
```

**Use Cases**:
- Small to medium datasets (< 10GB)
- Fast random access needed
- Data fits in RAM

#### FileDataset (Placeholder)

```mojo
from shared.data import FileDataset

# Define file paths
var file_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
var dataset = FileDataset(file_paths)

# Note: File loading not yet implemented
# Use TensorDataset with preprocessed data instead
```

**Current Status**: API defined, file loading deferred

**Workaround**: Preprocess files in Python, load as tensors

### 2. Data Loaders

#### DataLoader

```mojo
from shared.data import TensorDataset, DataLoader

# Create dataset
var dataset = TensorDataset(data, labels)

# Create loader with batching
var loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,     # Randomize order
    seed=42          # For reproducibility
)

# Iterate over batches
for batch in loader:
    print("Batch size:", batch.size)
    print("Data shape:", batch.data.shape)
    print("Labels:", batch.labels)
```

**Features**:
- Automatic batching
- Optional shuffling with seed control
- Handles partial final batch correctly
- Iterator protocol for easy loops

**Parameters**:
- `dataset`: Dataset to load from
- `batch_size`: Number of samples per batch
- `shuffle`: Whether to randomize order (default: False)
- `seed`: Random seed for reproducible shuffling (default: 42)

### 3. Sampling Strategies

#### SequentialSampler

```mojo
from shared.data import SequentialSampler

# Sequential access: 0, 1, 2, 3, ...
var sampler = SequentialSampler(num_samples=100)

for index in sampler:
    print(index)  # 0, 1, 2, ..., 99
```

#### RandomSampler

```mojo
from shared.data import RandomSampler

# Random access with seed
var sampler = RandomSampler(num_samples=100, seed=42)

for index in sampler:
    print(index)  # Random permutation of 0-99
```

### 4. Data Augmentation

See comprehensive augmentation documentation:

- [Image Augmentations](transforms.mojo)
- [Text Augmentations](text_transforms.mojo)
- [Generic Transforms](generic_transforms.mojo)

## Complete Workflows

### Training Pipeline with Augmentation

```mojo
from shared.data import (
    TensorDataset,
    DataLoader,
    RandomHorizontalFlip,
    RandomRotation,
    Normalize,
    Pipeline,
)

# 1. Create dataset
var train_data = load_training_data()
var train_labels = load_training_labels()
var dataset = TensorDataset(train_data, train_labels)

# 2. Setup augmentations
var transforms = List[Transform]()
transforms.append(RandomHorizontalFlip(0.5))
transforms.append(RandomRotation((15.0, 15.0)))
transforms.append(Normalize(0.5, 0.5))
var augmentations = Pipeline(transforms^)

# 3. Create data loader
var loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    seed=42
)

# 4. Training loop
for epoch in range(num_epochs):
    for batch in loader:
        # Apply augmentations
        var augmented_data = augmentations(batch.data)

        # Forward pass
        var predictions = model(augmented_data)

        # Compute loss
        var loss = criterion(predictions, batch.labels)

        # Backward pass and update
        loss.backward()
        optimizer.step()
```

### Validation Pipeline (No Augmentation)

```mojo
from shared.data import TensorDataset, DataLoader, Normalize

# Validation data (no random augmentations)
var val_data = load_validation_data()
var val_labels = load_validation_labels()
var val_dataset = TensorDataset(val_data, val_labels)

# Only apply normalization (no random transforms)
var normalize = Normalize(0.5, 0.5)

# No shuffling for validation
var val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# Validation loop
for batch in val_loader:
    var normalized_data = normalize(batch.data)
    var predictions = model(normalized_data)
    # Evaluate metrics
```

### Multi-Epoch Training

```mojo
from shared.data import TensorDataset, DataLoader

var dataset = TensorDataset(data, labels)
var loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Multiple epochs
for epoch in range(10):
    print("Epoch", epoch + 1)

    # Loader can be reused - creates new iterator each time
    for batch in loader:
        # Training step
        ...

    # Shuffling occurs fresh each epoch (with same seed)
```

## Best Practices

### 1. Dataset Size Considerations

**Small Datasets (< 1GB)**:
- Use TensorDataset
- Load everything into memory
- Fast random access

**Medium Datasets (1-10GB)**:
- Use TensorDataset if RAM available
- Otherwise preprocess in batches

**Large Datasets (> 10GB)**:
- Preprocess externally
- Load in batches
- Consider custom dataset implementation

### 2. Augmentation Guidelines

**Training**:
- Use random augmentations
- Set probability 0.3-0.5 for each transform
- Always normalize as final step

**Validation/Testing**:
- No random augmentations
- Only deterministic preprocessing (normalize, resize)
- Consistent with training preprocessing

### 3. Shuffling Strategy

**Training**: Always shuffle
```mojo
var loader = DataLoader(dataset, batch_size=32, shuffle=True, seed=42)
```

**Validation/Testing**: Don't shuffle
```mojo
var loader = DataLoader(dataset, batch_size=32, shuffle=False)
```

**Reproducibility**: Use consistent seed across experiments

### 4. Batch Size Selection

**Guidelines**:
- Powers of 2: 16, 32, 64, 128 (GPU friendly)
- Larger batches: More stable gradients, less noise
- Smaller batches: More updates per epoch, better generalization
- Typical range: 16-128 for most tasks

## API Reference

### Dataset Interface

```mojo
trait Dataset:
    fn __len__(self) -> Int
    fn __getitem__(self, index: Int) raises -> Tensor
```

### DataLoader

```mojo
struct DataLoader:
    fn __init__(
        out self,
        dataset: Dataset,
        batch_size: Int,
        shuffle: Bool = False,
        seed: Int = 42
    )
    fn __iter__(self) -> DataLoaderIterator
    fn __len__(self) -> Int
```

### Samplers

```mojo
trait Sampler:
    fn __iter__(self) -> Iterator[Int]
    fn __len__(self) -> Int

struct SequentialSampler(Sampler):
    fn __init__(out self, num_samples: Int)

struct RandomSampler(Sampler):
    fn __init__(out self, num_samples: Int, seed: Int = 42)
```

## Limitations and Future Work

### Current Limitations

1. **File Loading**: Not implemented (use TensorDataset with preprocessing)
2. **Memory Constraints**: TensorDataset requires data to fit in RAM
3. **No Streaming**: All data must be loaded upfront
4. **Limited Samplers**: Only sequential and random sampling

### Future Enhancements

1. **File Loading**:
   - Image decoders (JPEG, PNG, BMP)
   - NumPy binary format parser
   - CSV parsing utilities

2. **Advanced Sampling**:
   - Weighted sampling
   - Stratified sampling
   - Class-balanced sampling

3. **Performance**:
   - Multi-threaded data loading
   - Prefetching for pipeline efficiency
   - GPU-accelerated augmentations

4. **Streaming**:
   - Lazy loading for huge datasets
   - On-the-fly decompression
   - Network streaming

## Troubleshooting

### Problem: "File loading not implemented"

**Solution**: Use TensorDataset with preprocessed data

```python
# In Python: Preprocess and save
import numpy as np

# Load and preprocess
images = load_images()
processed = preprocess(images)
np.save("processed_data.npy", processed)
```

```mojo
// In Mojo: Load preprocessed data
var data = load_numpy_file("processed_data.npy")
var dataset = TensorDataset(data, labels)
```

### Problem: Memory overflow with TensorDataset

**Solution**: Reduce dataset size or implement batched preprocessing

```mojo
// Option 1: Subset of data
var dataset = TensorDataset(data[:10000], labels[:10000])

// Option 2: Process in chunks (custom implementation needed)
```

### Problem: Shuffling not reproducible

**Solution**: Use consistent seed

```mojo
// Set seed explicitly
var loader = DataLoader(dataset, batch_size=32, shuffle=True, seed=42)

// Same seed across runs produces same shuffle order
```

## See Also

- [Dataset Implementation](datasets.mojo)
- [Data Loader Implementation](loaders.mojo)
- [Augmentation Transforms](transforms.mojo)
- [Test Suite](../../tests/shared/data/)
```

## Success Criteria

- [ ] Public API exports are comprehensive
- [ ] Module documentation is complete
- [ ] Examples demonstrate all common workflows
- [ ] Best practices are documented
- [ ] Troubleshooting guide is helpful
- [ ] API reference is accurate
- [ ] Package is importable as `from shared.data import ...`

## References

### Related Issues

- Issue #428: [Plan] Data Utils
- Issue #429: [Test] Data Utils
- Issue #430: [Impl] Data Utils
- Issue #431: [Package] Data Utils (this issue)
- Issue #432: [Cleanup] Data Utils

### Implementation Files

- `/home/user/ml-odyssey/shared/data/__init__.mojo`
- `/home/user/ml-odyssey/shared/data/datasets.mojo`
- `/home/user/ml-odyssey/shared/data/loaders.mojo`
- `/home/user/ml-odyssey/shared/data/samplers.mojo`

## Implementation Notes

### Packaging Tasks

1. **Verify Exports**: Ensure `__init__.mojo` exports all public symbols
2. **Create README**: Comprehensive module documentation (shown above)
3. **Test Imports**: Validate all documented import patterns work
4. **Document Limitations**: Clear guidance on file loading workaround
5. **Provide Examples**: Complete workflow demonstrations

### Documentation Strategy

1. **Quick Start**: Simple examples first
2. **Complete Workflows**: Real-world training pipelines
3. **Best Practices**: Guidance for common scenarios
4. **Troubleshooting**: Solutions to likely problems
5. **API Reference**: Formal specifications

---

**Status**: Package structure defined, documentation to be created

**Last Updated**: 2025-11-19

**Prepared By**: Implementation Specialist (Level 3)
