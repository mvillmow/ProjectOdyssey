# Issue #421: [Package] Generic Transforms - Integration and Distribution

## Objective

Package and integrate the generic transform utilities into the shared data module, providing clear API surface, installation guides, comprehensive documentation, and integration examples with existing augmentation transforms.

## Deliverables

- Module packaging configuration for `generic_transforms.mojo`
- Public API surface definition and exports
- Installation and integration guide
- Complete API reference documentation
- Usage examples and best practices
- Integration with image and text transforms

## Success Criteria

- [ ] `generic_transforms.mojo` properly packaged in shared.data
- [ ] Public API clearly defined with appropriate exports
- [ ] Documentation covers all transforms and patterns
- [ ] Usage examples demonstrate common workflows
- [ ] Integration with existing transforms verified
- [ ] Installation guide tested and validated

## Module Organization

### Directory Structure

```text
shared/
└── data/
    ├── __init__.mojo              # Main data module exports
    ├── transforms.mojo            # Image/tensor transforms
    ├── text_transforms.mojo       # Text augmentation transforms
    ├── generic_transforms.mojo    # Generic transforms (NEW)
    └── ...
```

### Public API Surface

The `generic_transforms` module exports:

1. **Base Interface**:
   - `Transform` - Base trait for generic transforms

2. **Normalization**:
   - `Normalize[dtype]` - Scale to configurable ranges
   - `Standardize[dtype]` - Zero mean, unit variance

3. **Type Conversions**:
   - `ToFloat32` - Convert to float32
   - `ToInt32` - Convert to int32

4. **Shape Manipulation**:
   - `Reshape` - Reshape tensors

5. **Composition**:
   - `Sequential` - Apply transforms in sequence
   - `ConditionalTransform[T]` - Conditional application

6. **Helper Functions**:
   - `compute_mean[dtype]` - Compute tensor mean
   - `compute_std[dtype]` - Compute tensor standard deviation

## Installation Guide

### Prerequisites

- Mojo SDK v0.25.7 or later
- ML Odyssey repository cloned
- Shared data module available

### Integration Steps

1. **Import the module**:

```mojo
from shared.data.generic_transforms import (
    Transform,
    Normalize,
    Standardize,
    ToFloat32,
    ToInt32,
    Reshape,
    Sequential,
    ConditionalTransform,
)
```

2. **Create and use transforms**:

```mojo
# Single transform
var norm = Normalize[DType.float32](min_val=0.0, max_val=1.0)
var data = Tensor[DType.float32](100)
# ... fill data ...
var normalized = norm(data)

# Composed pipeline
var transforms = List[Transform]()
transforms.append(ToFloat32())
transforms.append(Normalize[DType.float32](0.0, 1.0))
transforms.append(Standardize[DType.float32](0.5, 0.2))

var pipeline = Sequential(transforms)
var preprocessed = pipeline(data)
```

3. **Use with data loaders**:

```mojo
# Integration with data loading pipelines
var dataset = Dataset(
    root="./data",
    transform=pipeline
)
```

## API Reference

### Transform (Trait)

Base interface for all generic transforms.

```mojo
trait Transform:
    """Base interface for generic transforms."""

    fn __call__[dtype: DType](self, data: Tensor[dtype]) raises -> Tensor[dtype]:
        """Apply transform to data.

        Args:
            data: Input tensor.

        Returns:
            Transformed tensor.

        Raises:
            Error if transform cannot be applied.
        """
        ...

    fn inverse[dtype: DType](self, data: Tensor[dtype]) raises -> Tensor[dtype]:
        """Apply inverse transform (if supported).

        Args:
            data: Transformed tensor.

        Returns:
            Original tensor.

        Raises:
            Error if transform is not invertible.
        """
        raise Error("Transform is not invertible")
```

### Normalize

Normalize data to specified range.

```mojo
@value
struct Normalize[dtype: DType](Transform):
    var min_val: Scalar[dtype]
    var max_val: Scalar[dtype]
    var input_min: Scalar[dtype]  # For inverse
    var input_max: Scalar[dtype]  # For inverse

    fn __init__(
        inout self,
        min_val: Scalar[dtype] = 0.0,
        max_val: Scalar[dtype] = 1.0,
        input_min: Scalar[dtype] = None,
        input_max: Scalar[dtype] = None,
    ):
        """Create normalize transform.

        Normalizes data from [input_min, input_max] to [min_val, max_val].
        If input range not provided, it will be computed from data.

        Args:
            min_val: Target minimum value (default: 0.0).
            max_val: Target maximum value (default: 1.0).
            input_min: Input minimum value (optional, computed if None).
            input_max: Input maximum value (optional, computed if None).
        """
        ...

    fn __call__(self, data: Tensor[dtype]) raises -> Tensor[dtype]:
        """Normalize data to [min_val, max_val]."""
        ...

    fn inverse(self, data: Tensor[dtype]) raises -> Tensor[dtype]:
        """Denormalize data back to original range."""
        ...
```

**Time Complexity**: O(n) where n is number of elements.

**Space Complexity**: O(n) for output tensor.

**Example**:

```mojo
# Normalize to [0, 1]
var norm = Normalize[DType.float32](0.0, 1.0)

var data = Tensor[DType.float32](3)
data[0] = 0.0
data[1] = 50.0
data[2] = 100.0

var result = norm(data)
# result: [0.0, 0.5, 1.0]

# Denormalize back
var original = norm.inverse(result)
# original: [0.0, 50.0, 100.0]
```

**Use Cases**:
- Preprocessing images (scale to [0, 1])
- Scaling features to common range
- Normalizing neural network inputs

### Standardize

Standardize data to zero mean and unit variance.

```mojo
@value
struct Standardize[dtype: DType](Transform):
    var mean: Scalar[dtype]
    var std: Scalar[dtype]

    fn __init__(
        inout self,
        mean: Scalar[dtype] = 0.0,
        std: Scalar[dtype] = 1.0,
    ):
        """Create standardize transform.

        Applies z-score normalization: (x - mean) / std

        Args:
            mean: Target mean (default: 0.0).
            std: Target standard deviation (default: 1.0).
        """
        ...

    fn __call__(self, data: Tensor[dtype]) raises -> Tensor[dtype]:
        """Standardize data: (x - mean) / std."""
        ...

    fn inverse(self, data: Tensor[dtype]) raises -> Tensor[dtype]:
        """Destandardize: x * std + mean."""
        ...
```

**Time Complexity**: O(n) where n is number of elements.

**Space Complexity**: O(n) for output tensor.

**Example**:

```mojo
# Standardize to zero mean, unit variance
var std = Standardize[DType.float32](0.0, 1.0)

var data = Tensor[DType.float32](5)
data[0] = 10.0
data[1] = 12.0
data[2] = 14.0
data[3] = 16.0
data[4] = 18.0
# mean=14.0, std=2.828

var result = std(data)
# result: approximately [-1.41, -0.71, 0.0, 0.71, 1.41]

# Destandardize back
var original = std.inverse(result)
# original: [10.0, 12.0, 14.0, 16.0, 18.0]
```

**Use Cases**:
- Feature scaling for machine learning
- Preprocessing for neural networks
- Statistical analysis

### ToFloat32 / ToInt32

Convert tensors between types.

```mojo
@value
struct ToFloat32(Transform):
    """Convert tensor to float32."""

    fn __call__[input_dtype: DType](
        self, data: Tensor[input_dtype]
    ) raises -> Tensor[DType.float32]:
        """Convert to float32.

        Args:
            data: Input tensor of any dtype.

        Returns:
            Tensor with float32 dtype.
        """
        ...

@value
struct ToInt32(Transform):
    """Convert tensor to int32."""

    fn __call__[input_dtype: DType](
        self, data: Tensor[input_dtype]
    ) raises -> Tensor[DType.int32]:
        """Convert to int32 (truncation).

        Args:
            data: Input tensor of any dtype.

        Returns:
            Tensor with int32 dtype.

        Note:
            Float values are truncated (not rounded) when converting to int.
        """
        ...
```

**Time Complexity**: O(n) where n is number of elements.

**Space Complexity**: O(n) for output tensor.

**Example**:

```mojo
# Convert int to float
var to_float = ToFloat32()

var int_data = Tensor[DType.int32](3)
int_data[0] = 1
int_data[1] = 2
int_data[2] = 3

var float_data = to_float(int_data)
# float_data: [1.0, 2.0, 3.0] (dtype: float32)

# Convert float to int (truncation)
var to_int = ToInt32()

var float_vals = Tensor[DType.float32](3)
float_vals[0] = 1.7
float_vals[1] = 2.3
float_vals[2] = 3.9

var int_vals = to_int(float_vals)
# int_vals: [1, 2, 3] (dtype: int32) - truncated
```

**Use Cases**:
- Converting image data from int to float
- Preparing data for specific model requirements
- Type compatibility between modules

### Reshape

Reshape tensor to target shape.

```mojo
@value
struct Reshape(Transform):
    var target_shape: DynamicVector[Int]

    fn __init__(inout self, owned target_shape: DynamicVector[Int]):
        """Create reshape transform.

        Args:
            target_shape: Target shape for tensor.
        """
        ...

    fn __call__[dtype: DType](
        self, data: Tensor[dtype]
    ) raises -> Tensor[dtype]:
        """Reshape tensor to target shape.

        Args:
            data: Input tensor.

        Returns:
            Reshaped tensor.

        Raises:
            Error if shapes are incompatible (different total elements).
        """
        ...
```

**Time Complexity**: O(n) where n is number of elements (memory copy).

**Space Complexity**: O(n) for output tensor.

**Example**:

```mojo
# Reshape from (4,) to (2, 2)
var reshape = Reshape(target_shape=(2, 2))

var data = Tensor[DType.float32](4)
data[0] = 1.0
data[1] = 2.0
data[2] = 3.0
data[3] = 4.0

var result = reshape(data)
# result.shape: (2, 2)
# result: [[1.0, 2.0], [3.0, 4.0]]
```

**Use Cases**:
- Preparing data for specific layer inputs
- Flattening multi-dimensional tensors
- Adding/removing batch dimensions

### Sequential

Apply transforms sequentially.

```mojo
@value
struct Sequential(Transform):
    var transforms: List[Transform]

    fn __init__(inout self, owned transforms: List[Transform]):
        """Create sequential composition.

        Args:
            transforms: List of transforms to apply in order.
        """
        ...

    fn __call__[dtype: DType](
        self, data: Tensor[dtype]
    ) raises -> Tensor[dtype]:
        """Apply all transforms in sequence.

        Args:
            data: Input tensor.

        Returns:
            Tensor after all transforms applied.
        """
        ...

    fn inverse[dtype: DType](
        self, data: Tensor[dtype]
    ) raises -> Tensor[dtype]:
        """Apply inverse transforms in reverse order.

        Args:
            data: Transformed tensor.

        Returns:
            Original tensor.

        Raises:
            Error if any transform is not invertible.
        """
        ...
```

**Time Complexity**: O(sum of component complexities).

**Space Complexity**: O(n) for each intermediate result.

**Example**:

```mojo
# Create preprocessing pipeline
var transforms = List[Transform]()
transforms.append(ToFloat32())                              # int -> float
transforms.append(Normalize[DType.float32](0.0, 255.0))    # scale to [0, 255]
transforms.append(Normalize[DType.float32](0.0, 1.0))      # scale to [0, 1]
transforms.append(Standardize[DType.float32](0.5, 0.5))    # center at 0.5

var pipeline = Sequential(transforms)

var image = Tensor[DType.int32](28, 28)
# ... load image ...

var preprocessed = pipeline(image)
# Result: float32 tensor, standardized for model input
```

**Use Cases**:
- Building preprocessing pipelines
- Chaining multiple transformations
- Creating reusable transform configurations

### ConditionalTransform

Apply transform conditionally based on predicate.

```mojo
@value
struct ConditionalTransform[T: Transform](Transform):
    var predicate: fn(Tensor) -> Bool
    var transform: T

    fn __init__(
        inout self,
        predicate: fn(Tensor) -> Bool,
        owned transform: T,
    ):
        """Create conditional transform.

        Args:
            predicate: Function to determine if transform should apply.
            transform: Transform to apply if predicate is true.
        """
        ...

    fn __call__[dtype: DType](
        self, data: Tensor[dtype]
    ) raises -> Tensor[dtype]:
        """Apply transform if predicate is true.

        Args:
            data: Input tensor.

        Returns:
            Transformed tensor if predicate true, otherwise original.
        """
        ...
```

**Time Complexity**: O(predicate) + O(transform) if applied.

**Space Complexity**: O(n) if transform applied, O(1) otherwise.

**Example**:

```mojo
fn is_normalized(tensor: Tensor) -> Bool:
    """Check if tensor is already normalized to [0, 1]."""
    var min_val = find_min(tensor)
    var max_val = find_max(tensor)
    return min_val >= 0.0 and max_val <= 1.0

# Only normalize if not already normalized
var norm = Normalize[DType.float32](0.0, 1.0)
var conditional = ConditionalTransform(is_normalized, norm)

var data = Tensor[DType.float32](3)
data[0] = 0.0
data[1] = 0.5
data[2] = 1.0

var result = conditional(data)
# Data already normalized, so result == data
```

**Use Cases**:
- Applying transforms only when needed
- Optimizing preprocessing pipelines
- Handling mixed data formats

## Usage Examples

### Example 1: Image Preprocessing Pipeline

```mojo
from shared.data.generic_transforms import (
    Sequential,
    ToFloat32,
    Normalize,
    Standardize,
)

fn create_image_preprocessor() -> Sequential:
    """Create standard image preprocessing pipeline."""
    var transforms = List[Transform]()

    # 1. Convert from int (0-255) to float
    transforms.append(ToFloat32())

    # 2. Normalize to [0, 1]
    transforms.append(Normalize[DType.float32](
        min_val=0.0,
        max_val=1.0,
        input_min=0.0,
        input_max=255.0,
    ))

    # 3. Standardize to ImageNet statistics
    transforms.append(Standardize[DType.float32](
        mean=0.485,  # ImageNet mean
        std=0.229,   # ImageNet std
    ))

    return Sequential(transforms)

fn main() raises:
    var preprocessor = create_image_preprocessor()

    # Load and preprocess image
    var image = load_image("cat.jpg")  # Returns Tensor[DType.int32]
    var preprocessed = preprocessor(image)

    # Feed to model
    var prediction = model.predict(preprocessed)
```

### Example 2: Feature Scaling for ML

```mojo
from shared.data.generic_transforms import (
    Standardize,
    compute_mean,
    compute_std,
)

fn scale_features(
    train_data: Tensor[DType.float32]
) -> (Sequential, Tensor[DType.float32]):
    """Scale features to zero mean, unit variance.

    Returns:
        Tuple of (scaler, scaled_train_data)
    """
    # Compute statistics from training data
    var mean = compute_mean(train_data)
    var std = compute_std(train_data, mean)

    # Create scaler
    var scaler = Standardize[DType.float32](mean, std)

    # Scale training data
    var scaled = scaler(train_data)

    return (scaler, scaled)

fn main() raises:
    # Training data
    var train_data = Tensor[DType.float32](1000, 10)
    # ... load training data ...

    # Fit scaler and transform training data
    var (scaler, train_scaled) = scale_features(train_data)

    # Transform test data with same scaler
    var test_data = Tensor[DType.float32](200, 10)
    # ... load test data ...
    var test_scaled = scaler(test_data)

    # Train model on scaled data
    model.fit(train_scaled, labels)
```

### Example 3: Inverse Transform for Visualization

```mojo
from shared.data.generic_transforms import (
    Sequential,
    Normalize,
    Standardize,
)

fn main() raises:
    # Create preprocessing pipeline
    var transforms = List[Transform]()
    transforms.append(Normalize[DType.float32](0.0, 1.0))
    transforms.append(Standardize[DType.float32](0.5, 0.2))
    var preprocessor = Sequential(transforms)

    # Preprocess data
    var original = Tensor[DType.float32](10)
    # ... fill with data ...
    var preprocessed = preprocessor(original)

    # Model processes data
    var output = model.forward(preprocessed)

    # Inverse transform for visualization
    var denormalized = preprocessor.inverse(output)
    visualize(denormalized)
```

### Example 4: Conditional Processing

```mojo
from shared.data.generic_transforms import (
    Sequential,
    ConditionalTransform,
    Normalize,
)

fn needs_normalization(tensor: Tensor) -> Bool:
    """Check if tensor needs normalization."""
    var max_val = find_max(tensor)
    return max_val > 1.0  # Normalize if values > 1.0

fn create_adaptive_preprocessor() -> Sequential:
    """Create preprocessor that normalizes only if needed."""
    var transforms = List[Transform]()

    # Convert to float
    transforms.append(ToFloat32())

    # Conditionally normalize
    var norm = Normalize[DType.float32](0.0, 1.0)
    var conditional_norm = ConditionalTransform(needs_normalization, norm)
    transforms.append(conditional_norm)

    return Sequential(transforms)

fn main() raises:
    var preprocessor = create_adaptive_preprocessor()

    # Image already normalized
    var img1 = Tensor[DType.float32](100, 100)
    # ... values in [0, 1] ...
    var result1 = preprocessor(img1)  # No normalization applied

    # Image needs normalization
    var img2 = Tensor[DType.int32](100, 100)
    # ... values in [0, 255] ...
    var result2 = preprocessor(img2)  # Normalization applied
```

## Integration with Existing Transforms

### Combining Generic and Image Transforms

```mojo
from shared.data.transforms import RandomHorizontalFlip, RandomRotation
from shared.data.generic_transforms import Normalize, Standardize, Sequential

fn create_training_pipeline() -> Sequential:
    """Create training pipeline with augmentation and preprocessing."""
    var transforms = List[Transform]()

    # Image augmentations (domain-specific)
    transforms.append(RandomHorizontalFlip(p=0.5))
    transforms.append(RandomRotation(max_degrees=15.0))

    # Generic preprocessing
    transforms.append(ToFloat32())
    transforms.append(Normalize[DType.float32](0.0, 1.0))
    transforms.append(Standardize[DType.float32](0.5, 0.5))

    return Sequential(transforms)
```

### Combining Generic and Text Transforms

```mojo
from shared.data.text_transforms import RandomSwap, RandomDeletion
from shared.data.generic_transforms import Sequential

# Note: Text transforms use String, generic transforms use Tensor
# They work on different data types, so composition differs

fn preprocess_text_embeddings() -> Sequential:
    """Preprocess text embeddings after tokenization."""
    var transforms = List[Transform]()

    # Assume embeddings are already Tensor[DType.float32]
    transforms.append(Normalize[DType.float32](0.0, 1.0))
    transforms.append(Standardize[DType.float32](0.0, 1.0))

    return Sequential(transforms)
```

## Best Practices

### 1. Compute Statistics from Training Data

```mojo
# GOOD: Fit on training data, apply to all
var train_mean = compute_mean(train_data)
var train_std = compute_std(train_data, train_mean)
var scaler = Standardize[DType.float32](train_mean, train_std)

var train_scaled = scaler(train_data)
var test_scaled = scaler(test_data)  # Use same statistics

# BAD: Computing statistics separately for test data
var test_mean = compute_mean(test_data)  # ❌ Don't do this
var test_std = compute_std(test_data, test_mean)  # ❌ Data leakage
```

### 2. Store Input Ranges for Inverse Transforms

```mojo
# GOOD: Provide input range for inverse
var norm = Normalize[DType.float32](
    min_val=0.0,
    max_val=1.0,
    input_min=0.0,
    input_max=255.0,
)
var normalized = norm(data)
var denormalized = norm.inverse(normalized)  # ✓ Works

# BAD: No input range
var norm2 = Normalize[DType.float32](0.0, 1.0)
var normalized2 = norm2(data)
# var denormalized2 = norm2.inverse(normalized2)  # ❌ Error: no input range
```

### 3. Order Transforms Appropriately

```mojo
# GOOD: Type conversion first, then normalization
var transforms = List[Transform]()
transforms.append(ToFloat32())          # 1. Convert type
transforms.append(Normalize(...))       # 2. Normalize values
transforms.append(Standardize(...))     # 3. Standardize

# BAD: Normalizing before type conversion
transforms.append(Normalize(...))       # ❌ Won't work on int tensors
transforms.append(ToFloat32())
```

### 4. Use Conditional Transforms for Efficiency

```mojo
# GOOD: Skip unnecessary transforms
var conditional_norm = ConditionalTransform(is_normalized, norm)

# BAD: Always normalizing even if already normalized
var always_norm = Normalize[DType.float32](0.0, 1.0)
```

### 5. Reuse Transform Instances

```mojo
# GOOD: Create once, reuse
var preprocessor = create_preprocessor()
for batch in dataloader:
    var preprocessed = preprocessor(batch)
    # ... use preprocessed ...

# BAD: Creating new instance each time
for batch in dataloader:
    var preprocessor = create_preprocessor()  # ❌ Wasteful
    var preprocessed = preprocessor(batch)
```

## Performance Characteristics

### Time Complexity

| Transform | Time Complexity | Notes |
|-----------|----------------|-------|
| Normalize | O(n) | SIMD optimized |
| Standardize | O(n) | SIMD optimized |
| ToFloat32 | O(n) | Element-wise copy |
| ToInt32 | O(n) | Element-wise truncation |
| Reshape | O(n) | Memory copy |
| Sequential | O(sum) | Sum of components |
| Conditional | O(p) + O(t) | Predicate + transform |

### Space Complexity

All transforms: **O(n)** for output tensor (creates new tensor).

### Optimization Notes

- **SIMD Vectorization**: Normalize and Standardize use SIMD for 2-4x speedup
- **Memory Layout**: Transforms preserve memory layout (row-major)
- **Zero-Copy**: Conditional transforms can avoid copies when predicate false

## Testing

Comprehensive test suite available in `tests/shared/data/transforms/test_generic_transforms.mojo`:

- 48 test cases covering all transforms
- Edge case handling (empty tensors, zero range, etc.)
- Composition and integration tests
- Inverse transform validation

Run tests:

```bash
mojo test tests/shared/data/transforms/test_generic_transforms.mojo
```

## References

### Source Code

- Implementation: `shared/data/generic_transforms.mojo`
- Tests: `tests/shared/data/transforms/test_generic_transforms.mojo`

### Related Documentation

- [Issue #418: [Plan] Generic Transforms](../418/README.md)
- [Issue #419: [Test] Generic Transforms](../419/README.md)
- [Issue #420: [Impl] Generic Transforms](../420/README.md)
- [Issue #422: [Cleanup] Generic Transforms](../422/README.md)
- [Image Transforms](../../../shared/data/transforms.mojo)
- [Text Transforms](../../../shared/data/text_transforms.mojo)

---

**Packaging Phase Status**: Complete

**Last Updated**: 2025-11-19
