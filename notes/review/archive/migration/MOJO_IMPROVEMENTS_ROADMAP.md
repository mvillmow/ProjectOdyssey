# Mojo Improvements Roadmap

**Based on:** MOJO_CODEBASE_REVIEW.md
**Date:** 2025-11-22
**Current Version:** Mojo >=0.25.7.0.dev2025110405

## Quick Action Matrix

| Priority | Action | Impact | Effort | Files Affected |
|----------|--------|--------|--------|----------------|
| 🔴 HIGH | SIMD Optimization | 2-8x perf | 2-3 weeks | shared/core/{arithmetic,elementwise}.mojo |
| 🔴 HIGH | Gradient Checking | Bug prevention | 1 week | tests/shared/core/test_*_backward.mojo |
| 🟡 MEDIUM | GPU Acceleration | 10-100x perf | 4-6 weeks | shared/core/gpu/ (new) |
| 🟡 MEDIUM | Type Specialization | 10-30% perf | 2-3 weeks | shared/core/types/typed_tensor.mojo (new) |
| 🟡 MEDIUM | Array API Complete | Feature complete | 2-3 weeks | shared/core/extensor.mojo |
| 🟢 LOW | Expand Traits | Better abstractions | 1-2 weeks | shared/core/traits.mojo (new) |
| 🟢 LOW | Property Testing | Edge cases | 1-2 weeks | tests/shared/core/test_properties.mojo (new) |

---

## 🔴 HIGH Priority Actions

### 1. SIMD Optimization for Core Operations

**Current State:**

- Manual loops in arithmetic operations (add, subtract, multiply, divide)
- No vectorization in broadcasting
- Element-wise operations iterate one element at a time

**Target Performance:**

- 2-8x speedup for element-wise operations
- Reduced memory bandwidth usage
- Better CPU cache utilization

**Implementation Plan:**

#### Phase 1: Add SIMD Variants (Week 1)

**File:** `shared/core/arithmetic_simd.mojo` (new)

```mojo
from algorithm import vectorize
from sys.info import simdwidthof

fn add_simd[dtype: DType](a: ExTensor, b: ExTensor) raises -> ExTensor:
    """SIMD-optimized element-wise addition for same-shape tensors."""
    if a.shape() != b.shape():
        return add(a, b)  # Fall back to broadcasting

    var result = ExTensor(a.shape(), dtype)
    alias simd_width = simdwidthof[dtype]()

    @parameter
    fn vectorized_add[width: Int](idx: Int):
        var a_vec = a._data.bitcast[Scalar[dtype]]().simd_load[width](idx)
        var b_vec = b._data.bitcast[Scalar[dtype]]().simd_load[width](idx)
        result._data.bitcast[Scalar[dtype]]().simd_store[width](idx, a_vec + b_vec)

    vectorize[simd_width, vectorized_add](a.numel())
    return result^
```text

**Files to Update:**

1. `shared/core/arithmetic_simd.mojo` - New SIMD implementations
2. `shared/core/elementwise_simd.mojo` - SIMD for exp, log, sqrt, etc.
3. `shared/core/reduction_simd.mojo` - SIMD for sum, mean, max, min

#### Phase 2: Integration & Fallback (Week 2)

**File:** `shared/core/arithmetic.mojo`

```mojo
fn add(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Element-wise addition with automatic SIMD optimization."""
    # Use SIMD for same-shape, contiguous tensors
    if a.shape() == b.shape() and a.is_contiguous() and b.is_contiguous():
        @parameter
        if a.dtype() == DType.float32 or a.dtype() == DType.float64:
            return add_simd(a, b)

    # Fall back to broadcasting for different shapes
    # ... existing broadcasting implementation
```text

#### Phase 3: Benchmarking & Tuning (Week 3)

**File:** `benchmarks/bench_simd_operations.mojo` (new)

```mojo
from benchmark import Bench, BenchConfig, BenchResult
from shared.core import ExTensor, zeros, ones
from shared.core.arithmetic import add
from shared.core.arithmetic_simd import add_simd

fn bench_add_scalar() raises -> BenchResult:
    """Benchmark scalar addition."""
    var a = ones([1024, 1024], DType.float32)
    var b = ones([1024, 1024], DType.float32)

    @parameter
    fn run_add():
        _ = add(a, b)

    return Bench.run[run_add]()

fn bench_add_simd() raises -> BenchResult:
    """Benchmark SIMD addition."""
    var a = ones([1024, 1024], DType.float32)
    var b = ones([1024, 1024], DType.float32)

    @parameter
    fn run_add():
        _ = add_simd(a, b)

    return Bench.run[run_add]()
```text

**Success Criteria:**

- [ ] 2x speedup for float32 operations on 1024x1024 tensors
- [ ] 4x speedup for float32 operations on aligned data
- [ ] All existing tests pass
- [ ] Numerical accuracy maintained (< 1e-6 difference)

---

### 2. Comprehensive Gradient Checking

**Current State:**

- Backward passes implemented but limited validation
- No systematic gradient checking tests
- Risk of gradient bugs in complex operations

**Target State:**

- Automated gradient checking for all backward passes
- Numerical validation against finite differences
- Regression tests for gradient correctness

**Implementation Plan:**

#### Phase 1: Gradient Checking Utility (Days 1-2)

**File:** `tests/shared/utils/gradient_checker.mojo` (new)

```mojo
from shared.core import ExTensor, zeros_like
from math import abs

fn check_gradients(
    forward_fn: fn(ExTensor) -> ExTensor,
    backward_fn: fn(ExTensor, ExTensor) -> ExTensor,
    input: ExTensor,
    epsilon: Float64 = 1e-5,
    tolerance: Float64 = 1e-3
) raises -> Bool:
    """Verify gradients using finite differences.

    Args:
        forward_fn: Forward pass function
        backward_fn: Backward pass function
        input: Input tensor for testing
        epsilon: Step size for finite differences
        tolerance: Maximum allowed difference

    Returns:
        True if gradients are correct within tolerance
    """
    # Compute analytical gradient
    var output = forward_fn(input)
    var grad_output = zeros_like(output)
    # Set grad_output to 1.0 for all elements
    for i in range(output.numel()):
        grad_output._set_float64(i, 1.0)

    var analytical_grad = backward_fn(grad_output, input)

    # Compute numerical gradient using finite differences
    var numerical_grad = zeros_like(input)
    for i in range(input.numel()):
        # f(x + epsilon)
        var input_plus = input.copy()
        var val = input_plus._get_float64(i)
        input_plus._set_float64(i, val + epsilon)
        var output_plus = forward_fn(input_plus)
        var sum_plus = output_plus.sum()

        # f(x - epsilon)
        var input_minus = input.copy()
        input_minus._set_float64(i, val - epsilon)
        var output_minus = forward_fn(input_minus)
        var sum_minus = output_minus.sum()

        # Gradient: (f(x+e) - f(x-e)) / (2*e)
        var grad = (sum_plus - sum_minus) / (2.0 * epsilon)
        numerical_grad._set_float64(i, grad)

    # Compare gradients
    var max_diff = 0.0
    for i in range(input.numel()):
        var analytical = analytical_grad._get_float64(i)
        var numerical = numerical_grad._get_float64(i)
        var diff = abs(analytical - numerical)
        if diff > max_diff:
            max_diff = diff

    return max_diff < tolerance
```text

#### Phase 2: Add Tests for All Backward Passes (Days 3-5)

**File:** `tests/shared/core/test_gradient_checking.mojo` (new)

```mojo
from tests.shared.utils.gradient_checker import check_gradients
from shared.core import ExTensor, zeros, ones
from shared.core.arithmetic import add, multiply
from shared.core.arithmetic_backward import add_backward, multiply_backward
from shared.core.activation import relu, relu_backward

fn test_add_gradient() raises:
    """Test add backward pass using gradient checking."""
    var a = ones([3, 4], DType.float32)

    fn forward(x: ExTensor) -> ExTensor:
        var b = ones([3, 4], DType.float32)
        return add(x, b)

    fn backward(grad_out: ExTensor, x: ExTensor) -> ExTensor:
        var b = ones([3, 4], DType.float32)
        var (grad_a, _) = add_backward(grad_out, x, b)
        return grad_a

    var passed = check_gradients(forward, backward, a)
    assert_true(passed, "Add gradient check failed")

fn test_multiply_gradient() raises:
    """Test multiply backward pass using gradient checking."""
    # ... similar structure

fn test_relu_gradient() raises:
    """Test ReLU backward pass using gradient checking."""
    # ... similar structure

fn test_conv2d_gradient() raises:
    """Test conv2d backward pass using gradient checking."""
    # ... similar structure

fn test_batch_norm_gradient() raises:
    """Test batch_norm backward pass using gradient checking."""
    # ... similar structure
```text

#### Phase 3: CI Integration (Days 6-7)

**File:** `.github/workflows/test-gradients.yml` (new)

```yaml
name: Gradient Checking Tests

on:
  pull_request:
    paths:
      - 'shared/core/**/*_backward.mojo'
      - 'tests/shared/core/test_gradient_checking.mojo'

jobs:
  gradient-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run gradient checking tests
        run: |
          pixi run mojo test tests/shared/core/test_gradient_checking.mojo
```text

**Success Criteria:**

- [ ] Gradient checking for all 20+ backward passes
- [ ] All gradients correct within 1e-3 tolerance
- [ ] CI catches gradient regressions automatically
- [ ] Documentation of gradient checking methodology

---

## 🟡 MEDIUM Priority Actions

### 3. GPU Acceleration

**Current State:**

- All operations run on CPU
- No GPU code in codebase
- GPU feature commented out in mojo.toml

**Target Performance:**

- 10-100x speedup for large tensor operations
- Support for training large models (ResNet-50, GPT-style)
- Automatic GPU/CPU fallback

**Implementation Plan:**

#### Phase 1: Infrastructure Setup (Weeks 1-2)

**File:** `shared/core/gpu/__init__.mojo` (new)

```mojo
"""GPU acceleration module for ML Odyssey.

Provides hardware-agnostic GPU operations using Mojo's gpu package.
"""

from gpu import DeviceArray, is_gpu_available
from shared.core import ExTensor

struct GPUConfig:
    """GPU configuration and device management."""
    var device_id: Int
    var auto_transfer: Bool

    fn __init__(inout self):
        self.device_id = 0
        self.auto_transfer = True

fn to_gpu(tensor: ExTensor) raises -> DeviceArray:
    """Transfer tensor to GPU memory."""
    if not is_gpu_available():
        raise Error("GPU not available")

    # Transfer data to GPU
    # ... implementation

fn from_gpu(device_array: DeviceArray) raises -> ExTensor:
    """Transfer GPU array back to CPU tensor."""
    # ... implementation
```text

**File:** `shared/core/gpu/matmul.mojo` (new)

```mojo
from gpu import launch_kernel, sync, DeviceArray

fn matmul_gpu(a: ExTensor, b: ExTensor) raises -> ExTensor:
    """GPU-accelerated matrix multiplication.

    Uses tiled matrix multiplication for optimal GPU performance.
    Falls back to CPU if GPU unavailable or tensors too small.
    """
    # Check GPU availability
    if not is_gpu_available():
        return matmul_cpu(a, b)

    # For small matrices, CPU is faster due to transfer overhead
    if a.numel() < 1024:
        return matmul_cpu(a, b)

    # Transfer to GPU
    var a_gpu = to_gpu(a)
    var b_gpu = to_gpu(b)
    var c_gpu = DeviceArray.zeros(a.shape[0], b.shape[1])

    # Launch tiled matmul kernel
    alias TILE_SIZE = 16

    @kernel
    fn matmul_kernel(row: Int, col: Int):
        var sum = 0.0
        for tile in range(0, a.shape[1], TILE_SIZE):
            # Tiled computation for memory efficiency
            for k in range(tile, min(tile + TILE_SIZE, a.shape[1])):
                sum += a_gpu[row, k] * b_gpu[k, col]
        c_gpu[row, col] = sum

    launch_kernel[matmul_kernel](a.shape[0], b.shape[1])
    sync()

    # Transfer back to CPU
    return from_gpu(c_gpu)
```text

#### Phase 2: Core Operations (Weeks 3-4)

Implement GPU versions of:

1. `matmul_gpu` - Matrix multiplication
2. `conv2d_gpu` - 2D convolution
3. `batch_norm_gpu` - Batch normalization
4. `relu_gpu` - ReLU activation
5. `softmax_gpu` - Softmax activation

#### Phase 3: Benchmarking & Optimization (Weeks 5-6)

**File:** `benchmarks/bench_gpu_operations.mojo` (new)

```mojo
fn benchmark_matmul_cpu_vs_gpu() raises:
    """Compare CPU vs GPU matmul performance."""
    for size in [128, 256, 512, 1024, 2048]:
        var a = ones([size, size], DType.float32)
        var b = ones([size, size], DType.float32)

        # CPU benchmark
        var cpu_result = bench_matmul_cpu(a, b)

        # GPU benchmark
        var gpu_result = bench_matmul_gpu(a, b)

        print(f"Size {size}x{size}: CPU={cpu_result.mean_time:.2f}ms, "
              f"GPU={gpu_result.mean_time:.2f}ms, "
              f"Speedup={cpu_result.mean_time / gpu_result.mean_time:.2f}x")
```text

**Success Criteria:**

- [ ] 10x speedup for 1024x1024 matmul
- [ ] 50x speedup for 2048x2048 matmul
- [ ] Automatic CPU fallback for small tensors
- [ ] < 1e-5 numerical difference vs CPU

---

### 4. Compile-Time Type Specialization

**Target:** Eliminate runtime type checks and enable more aggressive compiler optimizations.

**Implementation Plan:**

#### Phase 1: Typed Tensor Variants (Week 1)

**File:** `shared/core/types/typed_tensor.mojo` (new)

```mojo
struct TypedTensor[dtype: DType]:
    """Tensor with compile-time known dtype.

    Eliminates runtime dtype checks and enables more aggressive
    compiler optimizations.

    Parametric Features:
    - dtype: Compile-time data type (DType.float32, DType.float64, etc.)
    - Specialized operations for each dtype
    - Zero-cost type safety
    """
    var _data: UnsafePointer[Scalar[dtype]]
    var _shape: DynamicVector[Int]
    var _numel: Int

    fn __init__(inout self, shape: DynamicVector[Int]):
        """Initialize typed tensor with given shape."""
        self._shape = shape
        self._numel = 1
        for i in range(len(shape)):
            self._numel *= shape[i]

        # Allocate type-specific storage (no type erasure)
        self._data = UnsafePointer[Scalar[dtype]].alloc(self._numel)

    fn __getitem__(self, idx: Int) -> Scalar[dtype]:
        """Type-safe element access."""
        return self._data[idx]

    fn __setitem__(inout self, idx: Int, value: Scalar[dtype]):
        """Type-safe element assignment."""
        self._data[idx] = value

# Specialized addition for compile-time known types
fn add[dtype: DType](
    a: TypedTensor[dtype],
    b: TypedTensor[dtype]
) raises -> TypedTensor[dtype]:
    """Compile-time specialized addition.

    Compiler generates optimal code for each dtype without runtime checks.
    """
    if a._shape != b._shape:
        raise Error("Shape mismatch")

    var result = TypedTensor[dtype](a._shape)

    @parameter
    if dtype == DType.float32:
        # Specialized SIMD for float32
        alias simd_width = simdwidthof[DType.float32]()

        @parameter
        fn vectorized_add[width: Int](idx: Int):
            var a_vec = a._data.simd_load[width](idx)
            var b_vec = b._data.simd_load[width](idx)
            result._data.simd_store[width](idx, a_vec + b_vec)

        vectorize[simd_width, vectorized_add](a._numel)

    elif dtype == DType.float64:
        # Specialized SIMD for float64
        # ... similar implementation

    else:
        # Generic fallback
        for i in range(a._numel):
            result[i] = a[i] + b[i]

    return result^
```text

#### Phase 2: Fixed-Size Tensor Specialization (Week 2)

**File:** `shared/core/types/fixed_tensor.mojo` (new)

```mojo
struct FixedTensor[rows: Int, cols: Int, dtype: DType]:
    """Fixed-size tensor with compile-time known dimensions.

    Optimal for:
    - Convolution kernels (3x3, 5x5)
    - Small matrices (rotation matrices, etc.)
    - Embedded constants

    Benefits:
    - Stack allocation (no heap allocations)
    - Bounds checking at compile time
    - Maximum compiler optimization
    """
    var _data: StaticTuple[Scalar[dtype], rows * cols]

    fn __init__(inout self):
        """Initialize with zeros."""
        for i in range(rows * cols):
            self._data[i] = 0

    fn __getitem__(self, row: Int, col: Int) -> Scalar[dtype]:
        """Compile-time bounds-checked access."""
        constrained[row >= 0 and row < rows, "Row index out of bounds"]()
        constrained[col >= 0 and col < cols, "Col index out of bounds"]()
        return self._data[row * cols + col]

    fn matmul[other_cols: Int](
        self,
        other: FixedTensor[cols, other_cols, dtype]
    ) -> FixedTensor[rows, other_cols, dtype]:
        """Compile-time specialized matrix multiplication."""
        var result = FixedTensor[rows, other_cols, dtype]()

        @parameter
        for i in range(rows):
            @parameter
            for j in range(other_cols):
                var sum = Scalar[dtype](0)
                @parameter
                for k in range(cols):
                    sum += self[i, k] * other[k, j]
                result._data[i * other_cols + j] = sum

        return result^

# Example usage:
alias Conv3x3Kernel = FixedTensor[3, 3, DType.float32]
alias RotationMatrix = FixedTensor[3, 3, DType.float64]
```text

**Success Criteria:**

- [ ] 10-30% speedup for hot paths
- [ ] Zero runtime type checks
- [ ] Maintained backward compatibility with ExTensor
- [ ] Clear migration guide for typed tensors

---

### 5. Complete Array API Implementation

**Current State:**

- 150+ operations implemented
- Missing: reshape, advanced indexing, some matrix operations

**Target:** 100% Array API Standard 2023.12 compliance

**Implementation Plan:**

#### Week 1: Reshape & View Operations

**File:** `shared/core/shape_manipulation.mojo` (new)

```mojo
fn reshape(tensor: ExTensor, new_shape: DynamicVector[Int]) raises -> ExTensor:
    """Reshape tensor to new shape without copying data."""
    # Verify total elements match
    var old_numel = tensor.numel()
    var new_numel = 1
    for i in range(len(new_shape)):
        new_numel *= new_shape[i]

    if old_numel != new_numel:
        raise Error("Cannot reshape: element count mismatch")

    # Create view (share data)
    var result = ExTensor._create_view(tensor._data, new_shape, tensor.dtype())
    return result^
```text

#### Week 2-3: Advanced Indexing & Slicing

Add support for:

- Fancy indexing: `tensor[[0, 2, 5]]`
- Boolean indexing: `tensor[tensor > 0]`
- Ellipsis: `tensor[..., 0]`
- None indexing: `tensor[:, None, :]`

**Success Criteria:**

- [ ] Pass Array API Standard test suite
- [ ] 100% API coverage
- [ ] Documentation of all operations

---

## 🟢 LOW Priority Actions

### 6. Expand Trait System

**File:** `shared/core/traits.mojo` (new)

```mojo
trait Differentiable:
    """Types that support automatic differentiation."""
    fn forward(self, input: ExTensor) -> ExTensor: ...
    fn backward(self, grad_output: ExTensor) -> ExTensor: ...

trait Parameterized:
    """Types with learnable parameters."""
    fn parameters(self) -> List[ExTensor]: ...
    fn gradients(self) -> List[ExTensor]: ...
    fn zero_grad(inout self): ...

trait Serializable:
    """Types that can be saved/loaded."""
    fn save(self, path: String) raises: ...
    fn load(inout self, path: String) raises: ...

trait Composable:
    """Types that can be composed into pipelines."""
    fn compose[T: Composable](self, other: T) -> ComposedOp: ...
```text

---

### 7. Property-Based Testing

**File:** `tests/shared/core/test_properties.mojo` (new)

```mojo
from hypothesis import given, strategies as st

@given(st.tensor_strategy(min_dims=2, max_dims=4))
fn test_matmul_associativity(a: ExTensor, b: ExTensor, c: ExTensor) raises:
    """Test (AB)C = A(BC) for matrix multiplication."""
    var ab_c = matmul(matmul(a, b), c)
    var a_bc = matmul(a, matmul(b, c))
    assert_tensors_close(ab_c, a_bc, tolerance=1e-5)

@given(st.tensor_strategy())
fn test_add_commutative(a: ExTensor, b: ExTensor) raises:
    """Test a + b = b + a."""
    var ab = add(a, b)
    var ba = add(b, a)
    assert_tensors_equal(ab, ba)
```text

---

## Implementation Timeline

### Quarter 1 (Weeks 1-12)

**Weeks 1-3:** SIMD Optimization
**Weeks 4-5:** Gradient Checking
**Weeks 6-11:** GPU Acceleration
**Week 12:** Integration & Testing

### Quarter 2 (Weeks 13-24)

**Weeks 13-15:** Type Specialization
**Weeks 16-18:** Array API Completion
**Weeks 19-20:** Expanded Traits
**Weeks 21-22:** Property Testing
**Weeks 23-24:** Documentation & Polish

---

## Success Metrics

### Performance Targets

| Operation | Current | SIMD | GPU | Target |
|-----------|---------|------|-----|--------|
| Add (1024²) | 10ms | 2ms | 0.5ms | 20x |
| Matmul (1024²) | 500ms | 200ms | 10ms | 50x |
| Conv2d (256 3x3) | 2000ms | 800ms | 40ms | 50x |
| Training epoch | 60s | 30s | 3s | 20x |

### Quality Targets

- [ ] 100% gradient checking coverage
- [ ] 100% Array API compliance
- [ ] Zero memory leaks (Valgrind clean)
- [ ] < 1e-5 numerical accuracy vs reference
- [ ] 90%+ test coverage
- [ ] All CI tests passing

---

## Migration Guide

When implementing these improvements:

1. **Maintain backward compatibility**
   - Add new APIs alongside existing ones
   - Deprecate old APIs gradually
   - Provide migration scripts

2. **Benchmark before and after**
   - Use `benchmarks/` for all performance claims
   - Document speedups in commit messages
   - Add regression tests

3. **Document breaking changes**
   - Update CHANGELOG.md
   - Add migration guide to docs
   - Notify users in release notes

4. **Test thoroughly**
   - All existing tests must pass
   - Add new tests for new features
   - Run gradient checking on all backward passes

---

## Resources

- [Mojo SIMD Documentation](https://docs.modular.com/mojo/manual/types#simd-and-dtype)
- [Mojo GPU Programming](https://docs.modular.com/mojo/manual/gpu/intro-tutorial)
- [Array API Standard](https://data-apis.org/array-api/latest/)
- [Mojo Parameters](https://docs.modular.com/mojo/manual/parameters/)

**Maintainer:** ML Odyssey Team
**Last Updated:** 2025-11-22
