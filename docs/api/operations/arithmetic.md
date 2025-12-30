# Arithmetic Operations

Element-wise arithmetic operations on tensors with broadcasting support.

## Overview

All arithmetic operations support:

- **Broadcasting**: Automatic shape expansion following NumPy rules
- **Type safety**: Both operands must have the same dtype
- **Autograd**: Gradients tracked when recorded on tape

## Binary Operations

### Addition (+)

Element-wise addition.

```mojo
fn __add__(self, other: ExTensor) raises -> ExTensor
fn add(a: ExTensor, b: ExTensor) raises -> ExTensor
```

**Example:**

```mojo
from shared.core import zeros, ones, add

var a = ones[DType.float32](3, 4)
var b = ones[DType.float32](3, 4)
var c = a + b  # All elements are 2.0
var d = add(a, b)  # Same as a + b
```

**Broadcasting:**

```mojo
var x = randn[DType.float32](3, 4)  # Shape: (3, 4)
var y = randn[DType.float32](4)     # Shape: (4,)
var z = x + y  # y broadcasted to (3, 4)
```

### Subtraction (-)

Element-wise subtraction.

```mojo
fn __sub__(self, other: ExTensor) raises -> ExTensor
fn subtract(a: ExTensor, b: ExTensor) raises -> ExTensor
```

**Example:**

```mojo
var a = full(List[Int](3, 3), 5.0, DType.float32)
var b = ones[DType.float32](3, 3)
var c = a - b  # All elements are 4.0
```

### Multiplication (*)

Element-wise multiplication (Hadamard product).

```mojo
fn __mul__(self, other: ExTensor) raises -> ExTensor
fn multiply(a: ExTensor, b: ExTensor) raises -> ExTensor
```

**Example:**

```mojo
var a = full(List[Int](2, 3), 2.0, DType.float32)
var b = full(List[Int](2, 3), 3.0, DType.float32)
var c = a * b  # All elements are 6.0
```

### Division (/)

Element-wise division.

```mojo
fn __truediv__(self, other: ExTensor) raises -> ExTensor
fn divide(a: ExTensor, b: ExTensor) raises -> ExTensor
```

**Example:**

```mojo
var a = full(List[Int](2, 2), 10.0, DType.float32)
var b = full(List[Int](2, 2), 2.0, DType.float32)
var c = a / b  # All elements are 5.0
```

### Floor Division (//)

Element-wise floor division.

```mojo
fn __floordiv__(self, other: ExTensor) raises -> ExTensor
fn floor_divide(a: ExTensor, b: ExTensor) raises -> ExTensor
```

**Example:**

```mojo
var a = full(List[Int](2, 2), 7.0, DType.float32)
var b = full(List[Int](2, 2), 2.0, DType.float32)
var c = a // b  # All elements are 3.0
```

### Modulo (%)

Element-wise modulo.

```mojo
fn __mod__(self, other: ExTensor) raises -> ExTensor
fn modulo(a: ExTensor, b: ExTensor) raises -> ExTensor
```

**Example:**

```mojo
var a = full(List[Int](2, 2), 7.0, DType.float32)
var b = full(List[Int](2, 2), 3.0, DType.float32)
var c = a % b  # All elements are 1.0
```

### Power (**)

Element-wise exponentiation.

```mojo
fn __pow__(self, other: ExTensor) raises -> ExTensor
fn power(a: ExTensor, b: ExTensor) raises -> ExTensor
```

**Example:**

```mojo
var a = full(List[Int](2, 2), 2.0, DType.float32)
var b = full(List[Int](2, 2), 3.0, DType.float32)
var c = a ** b  # All elements are 8.0
```

## Unary Operations

### Negation (-)

Element-wise negation.

```mojo
fn __neg__(self) -> ExTensor
fn negative(x: ExTensor) raises -> ExTensor
```

**Example:**

```mojo
var a = ones[DType.float32](3, 3)
var b = -a  # All elements are -1.0
```

### Absolute Value

Element-wise absolute value.

```mojo
fn abs(self) raises -> ExTensor
fn absolute(x: ExTensor) raises -> ExTensor
```

**Example:**

```mojo
var a = full(List[Int](2, 2), -3.0, DType.float32)
var b = a.abs()  # All elements are 3.0
```

## Scalar Operations

Operations with scalar values on the right-hand side.

```mojo
var x = randn[DType.float32](3, 4)
var y = x + 1.0   # Add scalar
var z = x * 2.0   # Multiply by scalar
var w = x / 10.0  # Divide by scalar
var v = x ** 2.0  # Square all elements
```

## In-Place Operations

In-place operations modify the tensor directly (where supported).

```mojo
fn iadd(mut self, other: ExTensor) raises -> None
fn isub(mut self, other: ExTensor) raises -> None
fn imul(mut self, other: ExTensor) raises -> None
fn idiv(mut self, other: ExTensor) raises -> None
```

**Example:**

```mojo
var x = ones[DType.float32](3, 3)
x.iadd(ones[DType.float32](3, 3))  # x now contains 2.0
```

## Broadcasting Rules

Broadcasting automatically expands tensors to compatible shapes:

1. Shapes are aligned from the trailing dimensions
2. Dimensions of size 1 can broadcast to any size
3. Missing dimensions are treated as size 1

**Examples:**

```text
(3, 4) + (4,)     -> (3, 4)  # (4,) broadcasts to (1, 4) then (3, 4)
(3, 1) + (1, 4)   -> (3, 4)  # Both dimensions broadcast
(5, 3, 1) + (3, 4) -> (5, 3, 4)  # Leading dimension preserved
```

**Invalid broadcasting:**

```text
(3, 4) + (5,)     -> Error!  # 4 != 5, cannot broadcast
(3, 4) + (4, 3)   -> Error!  # Incompatible shapes
```

## Error Handling

Shape and type mismatches raise errors:

```mojo
var a = zeros[DType.float32](3, 4)
var b = zeros[DType.float32](5, 6)
var c = a + b  # Raises: "Cannot broadcast shapes [3, 4] and [5, 6]"

var x = zeros[DType.float32](3, 3)
var y = zeros[DType.int32](3, 3)
var z = x + y  # Raises: "DType mismatch: float32 vs int32"
```

## See Also

- [Reduction Operations](reduction.md) - Sum, mean, max, min
- [ExTensor Reference](../tensor.md) - Core tensor class
