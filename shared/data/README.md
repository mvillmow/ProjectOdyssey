# Data Processing Library(WIP)

The `data` library provides data loading, preprocessing, and augmentation utilities for training neural networks
in ML Odyssey. All components are implemented in Mojo for maximum performance.

## Purpose

This library provides:

- Dataset abstractions for various data sources
- High-performance data loaders with batching and shuffling
- Data transformation and augmentation pipelines
- Efficient memory management for large datasets

**Key Principle**: Provide flexible, composable data processing components that work seamlessly with the training
pipeline while maintaining high performance.

## Directory Organization

```text
data/
├── __init__.mojo           # Package root - exports main components
├── README.md               # This file
├── datasets.mojo           # Dataset abstractions and implementations
├── loaders.mojo            # Data loaders and samplers
└── transforms.mojo         # Data transformation and augmentation
```

## What Belongs in Data?

### Include

- Dataset abstractions (base traits/interfaces)
- Common dataset implementations (TensorDataset, ImageDataset)
- Data loaders with batching and shuffling
- Standard transforms (normalization, resizing, cropping)
- Data augmentation (random flips, rotations, crops)
- Sampling strategies (sequential, random, weighted)

### Exclude

- Paper-specific datasets (belongs in papers/\*/data/)
- Paper-specific data preprocessing (belongs with the paper)
- Custom file formats without broad applicability
- One-off data manipulation code

## Components

### Datasets (`datasets.mojo`)

Base abstractions and common implementations for datasets.

#### Dataset Trait

```mojo
trait Dataset:
    """Base interface for all datasets."""
    fn __len__(self) -> Int
    fn __getitem__(self, index: Int) -> (Tensor, Tensor)
```

#### TensorDataset

In-memory dataset for tensor data:

```mojo
struct TensorDataset(Dataset):
    """Dataset wrapping tensors."""
    var data: Tensor
    var targets: Tensor
    var transform: Optional[Transform]

    fn __init__(inout self, data: Tensor, targets: Tensor, transform: Optional[Transform] = None):
        """Create dataset from tensors."""
        self.data = data
        self.targets = targets
        self.transform = transform

    fn __len__(self) -> Int:
        """Return number of samples."""
        return self.data.shape[0]

    fn __getitem__(self, index: Int) -> (Tensor, Tensor):
        """Get sample at index."""
        var sample = self.data[index]
        var target = self.targets[index]

        if self.transform:
            sample = self.transform.value()(sample)

        return (sample, target)
```

**Use Case**: Training on MNIST, CIFAR-10, or any in-memory dataset

#### ImageDataset

Dataset for loading images from disk:

```mojo
struct ImageDataset(Dataset):
    """Dataset for loading images from files."""
    var image_paths: List[String]
    var labels: List[Int]
    var transform: Optional[Transform]

    fn __init__(inout self, image_paths: List[String], labels: List[Int], transform: Optional[Transform] = None):
        """Create image dataset from file paths."""
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    fn __len__(self) -> Int:
        return len(self.image_paths)

    fn __getitem__(self, index: Int) -> (Tensor, Tensor):
        """Load and transform image at index."""
        var image = load_image(self.image_paths[index])
        var label = self.labels[index]

        if self.transform:
            image = self.transform.value()(image)

        return (image, label)
```

**Use Case**: ImageNet, custom image datasets

#### CSVDataset

Dataset for loading data from CSV files:

```mojo
struct CSVDataset(Dataset):
    """Dataset for loading structured data from CSV."""
    var data: Tensor
    var targets: Tensor

    fn __init__(inout self, filepath: String, target_column: Int):
        """Load CSV file and split features from targets."""
        # Load CSV and convert to tensors
        pass
```

**Use Case**: Tabular data, benchmarks

### Data Loaders (`loaders.mojo`)

Efficient batch loading with parallel data loading and prefetching.

#### DataLoader

Main data loader with batching and shuffling:

```mojo
struct DataLoader:
    """Iterate over dataset in batches."""
    var dataset: Dataset
    var batch_size: Int
    var shuffle: Bool
    var drop_last: Bool
    var num_workers: Int

    fn __init__(
        inout self,
        dataset: Dataset,
        batch_size: Int = 1,
        shuffle: Bool = False,
        drop_last: Bool = False,
        num_workers: Int = 0
    ):
        """Initialize data loader."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers

    fn __iter__(self) -> DataLoaderIterator:
        """Return iterator over batches."""
        var sampler = RandomSampler(len(self.dataset)) if self.shuffle else SequentialSampler(len(self.dataset))
        var batch_sampler = BatchSampler(sampler, self.batch_size, self.drop_last)
        return DataLoaderIterator(self.dataset, batch_sampler)
```

**Features**:

- Batching: Group samples into batches
- Shuffling: Randomize sample order per epoch
- Drop last: Handle incomplete final batch
- Multi-worker: Parallel data loading (future)
- Prefetching: Overlap data loading with training (future)

#### Batch

Container for batch data:

```mojo
struct Batch:
    """Container for batch of data."""
    var inputs: Tensor
    var targets: Tensor
    var indices: List[Int]

    fn __init__(inout self, inputs: Tensor, targets: Tensor, indices: List[Int]):
        """Create batch container."""
        self.inputs = inputs
        self.targets = targets
        self.indices = indices
```

#### Samplers

Control sampling order:

```mojo
trait Sampler:
    """Base sampler interface."""
    fn __iter__(self) -> SamplerIterator
    fn __len__(self) -> Int

struct SequentialSampler(Sampler):
    """Sample elements sequentially."""
    var dataset_size: Int

struct RandomSampler(Sampler):
    """Sample elements randomly."""
    var dataset_size: Int
    var generator: RandomNumberGenerator

struct WeightedRandomSampler(Sampler):
    """Sample elements with given probabilities."""
    var weights: Tensor
    var num_samples: Int
```

### Transforms (`transforms.mojo`)

Data transformation and augmentation pipeline.

#### Transform Trait

```mojo
trait Transform:
    """Base interface for transforms."""
    fn __call__(self, x: Tensor) -> Tensor
```

#### Core Transforms

**ToTensor**: Convert data to tensor format

```mojo
struct ToTensor(Transform):
    """Convert input to Tensor."""
    fn __call__(self, x: Any) -> Tensor:
        """Convert to tensor."""
        return to_tensor(x)
```

**Normalize**: Normalize tensor values

```mojo
struct Normalize(Transform):
    """Normalize tensor with mean and std."""
    var mean: Float32
    var std: Float32

    fn __init__(inout self, mean: Float32, std: Float32):
        self.mean = mean
        self.std = std

    fn __call__(self, x: Tensor) -> Tensor:
        """Apply normalization: (x - mean) / std."""
        return (x - self.mean) / self.std
```

**Compose**: Chain multiple transforms

```mojo
struct Compose(Transform):
    """Compose multiple transforms."""
    var transforms: List[Transform]

    fn __init__(inout self, transforms: List[Transform]):
        self.transforms = transforms

    fn __call__(self, x: Tensor) -> Tensor:
        """Apply transforms sequentially."""
        var output = x
        for transform in self.transforms:
            output = transform(output)
        return output
```

#### Image Transforms

**Resize**: Resize images

```mojo
struct Resize(Transform):
    """Resize image to target size."""
    var height: Int
    var width: Int
    var interpolation: InterpolationMode

    fn __call__(self, x: Tensor) -> Tensor:
        """Resize image."""
        return resize_image(x, self.height, self.width, self.interpolation)
```

**CenterCrop**: Crop center region

```mojo
struct CenterCrop(Transform):
    """Crop center region of image."""
    var size: Int

    fn __call__(self, x: Tensor) -> Tensor:
        """Crop center."""
        return center_crop(x, self.size)
```

#### Data Augmentation

**RandomCrop**: Random crop for augmentation

```mojo
struct RandomCrop(Transform):
    """Randomly crop image."""
    var size: Int
    var padding: Int

    fn __call__(self, x: Tensor) -> Tensor:
        """Apply random crop."""
        var padded = pad_image(x, self.padding) if self.padding > 0 else x
        return random_crop(padded, self.size)
```

**RandomHorizontalFlip**: Random horizontal flip

```mojo
struct RandomHorizontalFlip(Transform):
    """Randomly flip image horizontally."""
    var p: Float32  # Probability

    fn __call__(self, x: Tensor) -> Tensor:
        """Apply random flip with probability p."""
        if random() < self.p:
            return horizontal_flip(x)
        return x
```

**RandomRotation**: Random rotation

```mojo
struct RandomRotation(Transform):
    """Randomly rotate image."""
    var degrees: (Float32, Float32)  # Angle range

    fn __call__(self, x: Tensor) -> Tensor:
        """Apply random rotation."""
        var angle = random_uniform(self.degrees[0], self.degrees[1])
        return rotate_image(x, angle)
```

## Usage Examples

### Basic Usage

```mojo
from shared.data import TensorDataset, DataLoader

# Create dataset
var dataset = TensorDataset(train_images, train_labels)

# Create loader
var loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=False
)

# Iterate over batches
for batch in loader:
    var inputs = batch.inputs
    var targets = batch.targets
    # ... training code
```

### With Transforms

```mojo
from shared.data import ImageDataset, DataLoader, Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip

# Create transform pipeline
var transform = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=0.5, std=0.5),
])

# Create dataset with transforms
var dataset = ImageDataset(image_paths, labels, transform=transform)

# Create loader
var loader = DataLoader(dataset, batch_size=128, shuffle=True)
```

### Custom Dataset

```mojo
from shared.data import Dataset, DataLoader

struct MyDataset(Dataset):
    """Custom dataset implementation."""
    var data: MyDataType

    fn __len__(self) -> Int:
        return len(self.data)

    fn __getitem__(self, index: Int) -> (Tensor, Tensor):
        # Custom loading logic
        var sample = self.data[index]
        var features = preprocess(sample)
        var label = extract_label(sample)
        return (features, label)

var dataset = MyDataset(my_data)
var loader = DataLoader(dataset, batch_size=32)
```

## Performance Considerations

### Memory Management

**Problem**: Loading large datasets can exhaust memory

**Solutions**:

1. **Lazy Loading**: Load data on-demand rather than all at once
2. **Memory Mapping**: Use memory-mapped files for large datasets
3. **Streaming**: Process data in streaming fashion without full load

### Data Loading Bottlenecks

**Problem**: Data loading can be slower than model computation

**Solutions**:

1. **Prefetching**: Load next batch while training current batch
2. **Multi-worker Loading**: Parallel data loading (future feature)
3. **Caching**: Cache frequently accessed data in memory

### Transform Performance

**Problem**: Complex transforms slow down data loading

**Solutions**:

1. **SIMD Optimization**: Vectorize transform operations
2. **Batch Transforms**: Apply transforms to entire batches
3. **Precompute**: Cache transformed data when possible

## Best Practices

### Dataset Design

1. **Keep It Simple**: Implement only essential loading logic in dataset
2. **Lazy Loading**: Don't load all data in `__init__`
3. **Validation**: Validate data integrity during loading
4. **Error Handling**: Gracefully handle missing or corrupted data

### Transform Design

1. **Composability**: Design transforms to be composable
2. **Determinism**: Support reproducible transforms with seed
3. **Type Safety**: Ensure transforms preserve tensor properties
4. **Documentation**: Document expected input/output formats

### Loader Configuration

1. **Batch Size**: Choose based on GPU memory and model size
2. **Shuffle**: Always shuffle training data
3. **Drop Last**: Drop incomplete batches for consistent shapes
4. **Num Workers**: Start with 0, increase if data loading is bottleneck

## Testing

Data processing components should be tested for:

1. **Correctness**: Transforms produce expected outputs
2. **Shape Preservation**: Batch shapes are consistent
3. **Memory Safety**: No leaks or corruption
4. **Performance**: Loading speed meets requirements
5. **Edge Cases**: Empty datasets, single samples, odd batch sizes

See `tests/shared/data/` for comprehensive test suite.

## Integration with Training

Data loaders integrate seamlessly with training loops:

```mojo
from shared.data import DataLoader
from shared.training import train_epoch, validate_epoch

# Create loaders
var train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
var val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    var train_loss = train_epoch(model, optimizer, train_loader, loss_fn)
    var val_loss = validate_epoch(model, val_loader, loss_fn)
```

## Future Enhancements

Planned features for future releases:

1. **Multi-worker Data Loading**: Parallel loading with worker processes
2. **Data Prefetching**: Overlap data loading with computation
3. **Distributed Data Loading**: Sharding for distributed training
4. **Advanced Samplers**: Stratified, cluster, importance sampling
5. **Caching Layer**: Intelligent caching of transformed data
6. **Streaming Datasets**: Support for infinite data streams

## References

- [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html)
- [TensorFlow Dataset API](https://www.tensorflow.org/guide/data)
- [Mojo Performance Guide](https://docs.modular.com/mojo/faq/#performance)

## Contributing

When adding new data processing components:

1. Follow the Dataset or Transform trait interface
2. Add comprehensive tests in `tests/shared/data/`
3. Document usage with examples
4. Optimize hot paths with SIMD
5. Update this README with new components
