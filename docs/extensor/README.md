# ExTensor: Extensible Tensor Library for Mojo

**Version**: 0.1.0
**License**: BSD-3-Clause

ExTensor is a high-performance tensor library for Mojo with NumPy-style broadcasting, designed for machine learning and scientific computing applications.

## Features

### Core Capabilities

- ✅ **Dynamic tensor shapes** - Runtime-determined dimensions
- ✅ **NumPy-style broadcasting** - Efficient element-wise operations on different-shaped tensors
- ✅ **Multiple data types** - Float32, Float64, Int32, Int64, and more
- ✅ **Zero-copy operations** - Stride-based broadcasting without data duplication
- ✅ **Type-safe API** - Compile-time type checking with Mojo

### Implemented Operations (57 total)

#### Arithmetic Operations (7) - All with Broadcasting ✨
- `add`, `subtract`, `multiply`, `divide`
- `floor_divide`, `modulo`, `power`

#### Creation Operations (7)
- `zeros`, `ones`, `full`, `empty`
- `arange`, `eye`, `linspace`

#### Comparison Operations (6)
- `equal`, `not_equal`
- `less`, `less_equal`
- `greater`, `greater_equal`

#### Element-wise Math (19)
- **Basic**: `abs`, `sign`, `exp`, `log`, `sqrt`
- **Trigonometric**: `sin`, `cos`, `tanh`
- **Rounding**: `ceil`, `floor`, `round`, `trunc`
- **Logical**: `logical_and`, `logical_or`, `logical_not`, `logical_xor`
- **Logarithmic**: `log2`, `log10`
- **Utilities**: `clip`

#### Matrix Operations (4)
- `matmul`, `transpose`, `dot`, `outer`

#### Reduction Operations (4)
- `sum`, `mean`, `max_reduce`, `min_reduce`

#### Shape Manipulation (8)
- `reshape`, `squeeze`, `unsqueeze`, `expand_dims`
- `flatten`, `ravel`, `concatenate`, `stack`

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/mvillmow/ml-odyssey.git
cd ml-odyssey

# Build the package (when Mojo package manager is available)
mojo package src/extensor -o extensor.mojopkg
```

### Using in Your Project

```mojo
from extensor import ExTensor, zeros, ones, add, multiply

fn main():
    # Create tensors
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 4

    let a = zeros(shape, DType.float32)
    let b = ones(shape, DType.float32)

    # Arithmetic with broadcasting
    let c = add(a, b)  # Element-wise addition
```

## Quick Start

### Creating Tensors

```mojo
from extensor import zeros, ones, full, arange, eye
from sys import DType

fn example_creation():
    # Create tensor filled with zeros
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 4
    let a = zeros(shape, DType.float32)  # 3x4 tensor of zeros

    # Create tensor filled with ones
    let b = ones(shape, DType.float64)  # 3x4 tensor of ones

    # Create tensor with specific value
    let c = full(shape, 5.0, DType.float32)  # 3x4 tensor of fives

    # Create range tensor
    let d = arange(0.0, 10.0, 1.0, DType.float32)  # [0, 1, 2, ..., 9]

    # Create identity matrix
    let e = eye(4, DType.float32)  # 4x4 identity matrix
```

### Arithmetic with Broadcasting

Broadcasting allows operations on tensors with different shapes, following NumPy rules:

```mojo
from extensor import ones, full, add, multiply
from sys import DType

fn example_broadcasting():
    # Same-shape arithmetic
    var shape_2d = DynamicVector[Int](2)
    shape_2d[0] = 3
    shape_2d[1] = 4

    let a = ones(shape_2d, DType.float32)      # Shape (3, 4)
    let b = full(shape_2d, 2.0, DType.float32) # Shape (3, 4)
    let c = add(a, b)  # Shape (3, 4), all 3.0

    # Scalar broadcasting
    var shape_scalar = DynamicVector[Int](0)  # Scalar
    let scalar = full(shape_scalar, 5.0, DType.float32)
    let d = multiply(a, scalar)  # Shape (3, 4), all 5.0

    # Vector-to-matrix broadcasting
    var shape_vec = DynamicVector[Int](1)
    shape_vec[0] = 4
    let vec = ones(shape_vec, DType.float32)  # Shape (4,)
    let e = add(a, vec)  # Shape (3, 4) - broadcasts along rows
```

### Shape Manipulation

```mojo
from extensor import arange, reshape, flatten, concatenate
from sys import DType

fn example_shapes():
    # Reshape tensor
    let a = arange(0.0, 12.0, 1.0, DType.float32)  # 12 elements
    var new_shape = DynamicVector[Int](2)
    new_shape[0] = 3
    new_shape[1] = 4
    let b = reshape(a, new_shape)  # Now 3x4

    # Flatten tensor
    let c = flatten(b)  # Back to 1D (12,)

    # Concatenate tensors
    var tensors = DynamicVector[ExTensor](2)
    tensors[0] = a
    tensors[1] = c
    let d = concatenate(tensors, axis=0)  # Concatenate along first axis
```

### Matrix Operations

```mojo
from extensor import zeros, matmul, transpose
from sys import DType

fn example_matrix():
    # Matrix multiplication
    var shape_a = DynamicVector[Int](2)
    shape_a[0] = 3
    shape_a[1] = 4
    var shape_b = DynamicVector[Int](2)
    shape_b[0] = 4
    shape_b[1] = 5

    let a = zeros(shape_a, DType.float32)  # 3x4
    let b = zeros(shape_b, DType.float32)  # 4x5
    let c = matmul(a, b)  # 3x5 result

    # Transpose
    let d = transpose(a)  # 4x3
```

## Broadcasting Rules

ExTensor follows NumPy-style broadcasting rules:

1. **Dimensions are compared right-to-left**
2. **Dimensions are compatible if**:
   - They are equal, OR
   - One of them is 1
3. **Missing dimensions are treated as 1**
4. **Output shape is the element-wise maximum**

### Broadcasting Examples

```text
Shape (3, 4, 5) + Shape (4, 5)    → (3, 4, 5)  # Missing dim treated as 1
Shape (3, 1, 5) + Shape (3, 4, 5) → (3, 4, 5)  # Size 1 broadcasts
Shape (3, 4)    + Shape (5,)      → ERROR      # Incompatible shapes
```

## API Reference

### Creation Operations

#### `zeros(shape, dtype) -> ExTensor`
Create a tensor filled with zeros.

**Parameters**:
- `shape: DynamicVector[Int]` - Shape of the output tensor
- `dtype: DType` - Data type of tensor elements

**Returns**: New ExTensor filled with zeros

#### `ones(shape, dtype) -> ExTensor`
Create a tensor filled with ones.

#### `full(shape, value, dtype) -> ExTensor`
Create a tensor filled with a specific value.

**Parameters**:
- `shape: DynamicVector[Int]` - Shape of the output tensor
- `value: Float64` - Fill value
- `dtype: DType` - Data type

#### `arange(start, stop, step, dtype) -> ExTensor`
Create a 1D tensor with evenly spaced values.

**Parameters**:
- `start: Float64` - Start value (inclusive)
- `stop: Float64` - Stop value (exclusive)
- `step: Float64` - Step size
- `dtype: DType` - Data type

### Arithmetic Operations

All arithmetic operations support NumPy-style broadcasting.

#### `add(a, b) -> ExTensor`
Element-wise addition with broadcasting.

**Parameters**:
- `a: ExTensor` - First tensor
- `b: ExTensor` - Second tensor

**Returns**: New tensor containing a + b

**Raises**: Error if shapes are not broadcast-compatible or dtypes don't match

#### `subtract(a, b) -> ExTensor`
Element-wise subtraction with broadcasting.

#### `multiply(a, b) -> ExTensor`
Element-wise multiplication with broadcasting.

#### `divide(a, b) -> ExTensor`
Element-wise division with broadcasting.

**Note**: Division by zero follows IEEE 754 semantics:
- `x / 0.0` where `x > 0` → `+inf`
- `x / 0.0` where `x < 0` → `-inf`
- `0.0 / 0.0` → `NaN`

#### `floor_divide(a, b) -> ExTensor`
Element-wise floor division with broadcasting.

#### `modulo(a, b) -> ExTensor`
Element-wise modulo operation with broadcasting.

#### `power(a, b) -> ExTensor`
Element-wise exponentiation with broadcasting.

### Shape Manipulation

#### `reshape(tensor, new_shape) -> ExTensor`
Reshape tensor to new shape.

**Parameters**:
- `tensor: ExTensor` - Input tensor
- `new_shape: DynamicVector[Int]` - New shape (total elements must match)

**Returns**: Reshaped tensor

#### `flatten(tensor) -> ExTensor`
Flatten tensor to 1D.

#### `concatenate(tensors, axis) -> ExTensor`
Concatenate tensors along existing axis.

**Parameters**:
- `tensors: DynamicVector[ExTensor]` - List of tensors to concatenate
- `axis: Int` - Axis along which to concatenate

#### `stack(tensors, axis) -> ExTensor`
Stack tensors along a new axis.

## Performance

### Broadcasting Optimization

ExTensor uses **stride-based broadcasting** which avoids unnecessary data copying:

- **Zero-copy broadcasting**: Operations work directly on views
- **Efficient indexing**: Computed indices use broadcast strides
- **No data duplication**: Memory efficient for large tensors

### Benchmarks

(Benchmarks to be added)

## Development Status

### Current Version: 0.1.0

**Completed** (Issues #219-220):
- ✅ Core arithmetic operations with broadcasting
- ✅ Broadcasting infrastructure
- ✅ Comprehensive test suite (355 tests)
- ✅ Shape manipulation operations
- ✅ Element-wise math operations

**In Progress**:
- ⏳ Packaging and distribution (#221)
- ⏳ Code cleanup and optimization (#222)

**Future Work**:
- ⬜ Additional Array API Standard operations
- ⬜ SIMD optimization
- ⬜ GPU acceleration
- ⬜ Automatic differentiation (autograd)

## Contributing

Contributions are welcome! Please see the main [ML Odyssey contributing guide](../../CONTRIBUTING.md).

## License

BSD-3-Clause - See [LICENSE](../../LICENSE) for details.

## Acknowledgments

- Inspired by NumPy, PyTorch, and JAX broadcasting semantics
- Built with [Mojo](https://www.modular.com/mojo) programming language
- Part of the [ML Odyssey](https://github.com/mvillmow/ml-odyssey) project

## Links

- **Homepage**: <https://github.com/mvillmow/ml-odyssey>
- **Issues**: <https://github.com/mvillmow/ml-odyssey/issues>
- **Documentation**: [docs/extensor/](.)
