# ExTensors Implementation Prompt

## Context and Motivation

You are tasked with designing and implementing **ExTensors** (Extensible Tensors), a comprehensive tensor class for the ML Odyssey project. This implementation will serve as the foundational data structure for all neural network operations in the codebase. ExTensors must be both flexible enough to handle dynamic workloads and optimized enough to achieve maximum performance when tensor shapes are known at compile-time.

### Why this matters:

- Tensors are the fundamental data structure in all deep learning frameworks
- Performance-critical ML workloads require compile-time optimization when possible
- Research implementations need flexibility to experiment with arbitrary tensor shapes
- Mojo's unique type system allows us to unify both dynamic and static tensor paths in a single elegant design

## Task Overview

Re-interpret GitHub issues #218-222 (originally scoped for basic arithmetic operations) to instead implement a complete, production-ready tensor class called **ExTensors**.

**Original scope:** Basic arithmetic operations (add, subtract, multiply, divide)
**New scope:** Complete tensor infrastructure with static/dynamic variants

## Technical Requirements

### 1. Dynamic Tensor System

Implement ExTensor as a runtime-flexible tensor class:

### ExTensor (Dynamic Tensor):

- Shape determined at runtime
- Flexible for research and experimentation
- Runtime shape validation with clear error messages
- Support for dynamic reshaping and shape inference
- Compatible with Python-style workflow
- Supports arbitrary dimensions (0D scalars to N-D tensors)
- Generic over data type via Mojo's DType system
- Efficient SIMD-optimized operations
- Memory-safe via Mojo's ownership system

### 2. Tensor Calculus Foundation

Implement operations grounded in tensor calculus principles:

### Core tensor algebra operations:

- **Tensor addition/subtraction:** Only defined for tensors of same rank and compatible dimensions
- **Tensor product (outer product):** Combines tensors of rank (m, n) to produce rank (m+n) tensor
- **Contraction:** Reduces tensor rank by summing over matched indices (Einstein summation)
- **Scalar multiplication:** Scales all tensor elements
- **Element-wise operations:** Apply functions element-wise while preserving tensor structure

### Type preservation:

- Operations should preserve or predictably transform tensor rank
- Clear semantics for covariant vs contravariant indices (future extension)
- Support for Einstein summation notation (future extension)

### 3. Complete Tensor API (Array API Standard 2024 + Mojo Built-ins)

Based on the **Array API Standard 2024**, **Mojo's built-in operations**, and PyTorch/NumPy conventions, implement this complete set of operations:

#### Creation Operations

- `zeros(shape, dtype)` - Create tensor filled with zeros
- `ones(shape, dtype)` - Create tensor filled with ones
- `full(shape, fill_value, dtype)` - Create tensor filled with a value
- `arange(start, stop, step, dtype)` - Create 1D tensor with evenly spaced values
- `from_array(data)` - Create tensor from array/list data
- `eye(n, m, k, dtype)` - Create 2D tensor with ones on diagonal
- `linspace(start, stop, num, dtype)` - Create 1D tensor with evenly spaced values
- `empty(shape, dtype)` - Create uninitialized tensor (fast)

#### Arithmetic Operations (Dunder Methods + Broadcasting)

**Binary Arithmetic** - Normal, reflected, and in-place variants:

- `add(a, b)` / `__add__` / `a + b` - Element-wise addition
- `subtract(a, b)` / `__sub__` / `a - b` - Element-wise subtraction
- `multiply(a, b)` / `__mul__` / `a * b` - Element-wise multiplication
- `divide(a, b)` / `__truediv__` / `a / b` - Element-wise true division
- `floor_divide(a, b)` / `__floordiv__` / `a // b` - Element-wise floor division
- `modulo(a, b)` / `__mod__` / `a % b` - Element-wise modulo
- `power(a, b)` / `__pow__` / `a ** b` - Element-wise exponentiation
- `matmul(a, b)` / `__matmul__` / `a @ b` - Matrix multiplication

**Reflected Variants** (for reverse operations like `2 + tensor`):

- `__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__`, `__rfloordiv__`, `__rmod__`, `__rpow__`, `__rmatmul__`

**In-place Variants** (for operations like `a += b`):

- `__iadd__`, `__isub__`, `__imul__`, `__itruediv__`, `__ifloordiv__`, `__imod__`, `__ipow__`, `__imatmul__`

### Unary Operations

- `negative(a)` / `__neg__` / `-a` - Element-wise negation
- `positive(a)` / `__pos__` / `+a` - Element-wise positive (identity)
- `abs(a)` / `__abs__` - Element-wise absolute value
- `invert(a)` / `__invert__` / `~a` - Bitwise NOT (for integer/bool tensors)

#### Bitwise Operations (for integer and bool tensors)

**Binary Bitwise** - Normal, reflected, and in-place variants:

- `bitwise_and(a, b)` / `__and__` / `a & b` - Bitwise AND
- `bitwise_or(a, b)` / `__or__` / `a | b` - Bitwise OR
- `bitwise_xor(a, b)` / `__xor__` / `a ^ b` - Bitwise XOR
- `left_shift(a, b)` / `__lshift__` / `a << b` - Left shift
- `right_shift(a, b)` / `__rshift__` / `a >> b` - Right shift

### Reflected Bitwise

- `__rand__`, `__ror__`, `__rxor__`, `__rlshift__`, `__rrshift__`

### In-place Bitwise

- `__iand__`, `__ior__`, `__ixor__`, `__ilshift__`, `__irshift__`

#### Comparison Operations (return bool tensors)

- `equal(a, b)` / `__eq__` / `a == b` - Element-wise equality
- `not_equal(a, b)` / `__ne__` / `a != b` - Element-wise inequality
- `less(a, b)` / `__lt__` / `a < b` - Element-wise less-than
- `less_equal(a, b)` / `__le__` / `a <= b` - Element-wise less-or-equal
- `greater(a, b)` / `__gt__` / `a > b` - Element-wise greater-than
- `greater_equal(a, b)` / `__ge__` / `a >= b` - Element-wise greater-or-equal

#### Pointwise Math Operations (Element-wise)

### Trigonometric Functions

- `sin(a)` - Element-wise sine
- `cos(a)` - Element-wise cosine
- `tan(a)` - Element-wise tangent
- `asin(a)` - Element-wise arcsine
- `acos(a)` - Element-wise arccosine
- `atan(a)` - Element-wise arctangent
- `atan2(a, b)` - Element-wise 2-argument arctangent

### Hyperbolic Functions

- `sinh(a)` - Element-wise hyperbolic sine
- `cosh(a)` - Element-wise hyperbolic cosine
- `tanh(a)` - Element-wise hyperbolic tangent
- `asinh(a)` - Element-wise inverse hyperbolic sine
- `acosh(a)` - Element-wise inverse hyperbolic cosine
- `atanh(a)` - Element-wise inverse hyperbolic tangent

### Exponential and Logarithmic

- `exp(a)` - Element-wise exponential (e^x)
- `exp2(a)` - Element-wise base-2 exponential (2^x)
- `expm1(a)` - Element-wise exp(x) - 1 (accurate for small x)
- `log(a)` - Element-wise natural logarithm
- `log2(a)` - Element-wise base-2 logarithm
- `log10(a)` - Element-wise base-10 logarithm
- `log1p(a)` - Element-wise log(1 + x) (accurate for small x)

### Power and Root Functions

- `sqrt(a)` - Element-wise square root
- `cbrt(a)` - Element-wise cube root
- `square(a)` - Element-wise square (x^2)
- `rsqrt(a)` - Element-wise reciprocal square root (1/sqrt(x))

**Rounding Functions** (Dunder Methods):

- `ceil(a)` / `__ceil__` - Element-wise ceiling
- `floor(a)` / `__floor__` - Element-wise floor
- `trunc(a)` / `__trunc__` - Element-wise truncation toward zero
- `round(a)` / `__round__` - Element-wise rounding to nearest integer

### Other Pointwise Operations

- `sign(a)` - Element-wise sign (-1, 0, +1)
- `copysign(a, b)` - Element-wise copy sign from b to |a|
- `fma(a, b, c)` - Fused multiply-add: a*b + c
- `clip(a, min, max)` - Clamp values to range [min, max]
- `reciprocal(a)` - Element-wise reciprocal (1/x)

#### Matrix Operations

- `matmul(a, b)` / `a @ b` - Matrix multiplication
- `transpose(a, axes=None)` - Transpose tensor along specified axes
- `dot(a, b)` - Dot product (1D) or matrix product
- `outer(a, b)` - Outer product of vectors
- `inner(a, b)` - Inner product of tensors
- `tensordot(a, b, axes)` - Tensor contraction along specified axes

#### Reduction Operations

- `sum(a, axis=None, keepdims=False)` - Sum along axis
- `prod(a, axis=None, keepdims=False)` - Product along axis
- `mean(a, axis=None, keepdims=False)` - Mean along axis
- `var(a, axis=None, keepdims=False)` - Variance along axis
- `std(a, axis=None, keepdims=False)` - Standard deviation along axis
- `max(a, axis=None, keepdims=False)` - Maximum along axis
- `min(a, axis=None, keepdims=False)` - Minimum along axis
- `argmax(a, axis=None)` - Index of maximum along axis
- `argmin(a, axis=None)` - Index of minimum along axis
- `count_nonzero(a, axis=None)` - Count non-zero elements (Array API 2024)
- `cumulative_sum(a, axis)` - Cumulative sum along axis
- `cumulative_prod(a, axis)` - Cumulative product (Array API 2024)
- `all(a, axis=None, keepdims=False)` - Test if all elements are True
- `any(a, axis=None, keepdims=False)` - Test if any element is True

#### Shape Manipulation

- `reshape(a, new_shape)` - Reshape tensor
- `squeeze(a, axis=None)` - Remove dimensions of size 1
- `unsqueeze(a, axis)` / `expand_dims(a, axis)` - Add dimension of size 1
- `flatten(a)` - Flatten to 1D
- `ravel(a)` - Return flattened view (zero-copy if possible)
- `concatenate(tensors, axis)` - Join tensors along existing axis
- `stack(tensors, axis)` - Stack tensors along new axis
- `split(a, indices_or_sections, axis)` - Split tensor into multiple sub-tensors
- `tile(a, reps)` - Repeat tensor along axes
- `repeat(a, repeats, axis)` - Repeat elements along axis
- `broadcast_to(a, shape)` - Broadcast tensor to new shape
- `permute(a, axes)` - Permute tensor dimensions

#### Indexing and Slicing

- `__getitem__(indices)` / `a[i, j, k]` - Tensor indexing
- `__setitem__(indices, value)` / `a[i, j, k] = value` - Tensor assignment
- `take(a, indices, axis)` - Take elements along axis
- `take_along_axis(a, indices, axis)` - Gather along axis (Array API 2024)
- `put(a, indices, values)` - Put values at indices
- `gather(a, dim, index)` - Gather values along dimension
- `scatter(a, dim, index, src)` - Scatter values along dimension
- `where(condition, x, y)` - Select elements from x or y based on condition
- `masked_select(a, mask)` - Select elements where mask is True

#### Utility and Inspection Operations

- `copy(a)` / `clone(a)` - Create deep copy
- `diff(a, n=1, axis=-1)` - Discrete difference (Array API 2024)
- `__len__` - Return size of first dimension
- `__bool__` - Convert single-element tensor to bool
- `__int__` - Convert single-element tensor to int
- `__float__` - Convert single-element tensor to float
- `__str__` - String representation
- `__repr__` - Detailed representation
- `__hash__` - Hash for hashable tensors (if immutable)
- `__contains__` - Check if value is in tensor
- `__divmod__(a, b)` - Return (quotient, remainder) tuple
- `item()` - Extract scalar value (for single-element tensors)
- `tolist()` - Convert to nested Python list
- `numel()` - Total number of elements
- `dim()` - Number of dimensions
- `size(axis=None)` - Shape of tensor or size along axis
- `stride(axis=None)` - Stride of tensor or stride along axis
- `is_contiguous()` - Check if tensor has contiguous memory layout
- `contiguous()` - Return contiguous copy if not already contiguous

### 4. Data Type Support

Support for arbitrary data types via Mojo's DType system:

### Initial supported types:

- Float: `DType.float16`, `DType.float32`, `DType.float64`
- Integer: `DType.int8`, `DType.int16`, `DType.int32`, `DType.int64`
- Unsigned: `DType.uint8`, `DType.uint16`, `DType.uint32`, `DType.uint64`
- Boolean: `DType.bool`

### Extension path:

- Custom data types via parametric DType parameter
- Mixed-precision operations (future)
- Complex numbers (future)

### 5. Dimension Support

Support for arbitrary number of dimensions (0D scalars to N-D tensors):

- **0D tensors:** Scalars with shape `()`
- **1D tensors:** Vectors
- **2D tensors:** Matrices
- **3D+ tensors:** Higher-rank tensors (images, batches, sequences)
- **Upper limit:** Practical limit of 8-16 dimensions (most ML uses ≤5)

### 6. Broadcasting Support

Implement NumPy-style broadcasting rules:

1. Compare shapes element-wise from right to left
1. Dimensions are compatible if they are equal or one is 1
1. Missing dimensions treated as 1
1. Output shape is element-wise maximum of input shapes

### Examples:

```text
(3, 4, 5) + (4, 5)    → (3, 4, 5)  # Missing dimension = 1
(3, 1, 5) + (3, 4, 5) → (3, 4, 5)  # Size 1 broadcasts
(3, 4, 5) + (3, 4, 1) → (3, 4, 5)  # Size 1 broadcasts
```text

### 7. Memory and Performance

### Memory layout:

- Row-major (C-order) as default
- Strided memory layout for efficient slicing
- Contiguous memory guarantee for SIMD operations

### Performance optimizations:

- SIMD vectorization for element-wise operations
- Compile-time shape optimization for static tensors
- Cache-friendly memory access patterns
- Lazy evaluation for operation fusion (future)

### 8. Error Handling and Safety

### Mojo's safety-first approach:

- Runtime shape validation with clear error messages
- Type safety via DType parametric types
- Memory safety via ownership system (owned, borrowed, inout)
- Clear error messages for dimension mismatches
- IEEE 754 semantics for floating-point edge cases:
  - `x / 0.0` where `x > 0` → `+inf`
  - `x / 0.0` where `x < 0` → `-inf`
  - `0.0 / 0.0` → `NaN`

### Edge case handling:

- Empty tensors (0-element shapes)
- Large value overflow → saturate to infinity
- Zero-dimensional scalar tensors
- Non-broadcastable shape errors

## Implementation Approach

Follow the 5-phase development workflow:

### Phase 1: Plan (Issue #218)

**Objective:** Design ExTensor architecture and API specification

### Deliverables:

1. Complete API specification document
1. Memory layout specification
1. Broadcasting algorithm specification
1. Data type support matrix
1. Performance optimization strategy
1. Error handling specification
1. Shape representation and ownership model

### Success criteria:

- Unambiguous API contracts for all operations
- Comprehensive edge case documentation
- Performance optimization strategy defined
- Design ready for Test and Implementation phases

### Phase 2: Test (Issue #219) - Parallel after Plan

**Objective:** Write comprehensive test suite following TDD principles

### Test categories:

1. **Creation tests:** All creation operations with various dtypes and shapes
1. **Arithmetic tests:** Element-wise ops with broadcasting, edge cases
1. **Matrix operation tests:** Matmul, transpose with various dimensions
1. **Reduction tests:** All reductions with keepdims, axis variations
1. **Shape manipulation tests:** Reshape, squeeze, expand, concatenate
1. **Indexing tests:** Slicing, advanced indexing, assignment
1. **Broadcasting tests:** All broadcasting rule combinations
1. **Edge case tests:** Empty tensors, scalars, overflow, NaN, inf
1. **Type tests:** All supported dtypes
1. **Performance tests:** SIMD operation benchmarks

### Success criteria:

- 100% coverage of API surface
- All tests pass with clear error messages
- Performance benchmarks demonstrate SIMD optimization

### Phase 3: Implementation (Issue #220) - Parallel after Plan

**Objective:** Implement ExTensor following TDD

### Implementation order:

1. Core ExTensor struct definition
1. Memory allocation and layout (row-major, strided)
1. Creation operations
1. Basic arithmetic (no broadcasting)
1. Broadcasting infrastructure
1. Arithmetic with broadcasting
1. Matrix operations
1. Reduction operations
1. Shape manipulation
1. Indexing and slicing
1. Comparison operations
1. Utility operations

### Success criteria:

- All tests pass
- SIMD optimizations applied where possible
- Memory safety verified
- Performance meets benchmark targets

### Phase 4: Package (Issue #221) - Parallel after Plan

**Objective:** Package ExTensors for distribution and reuse

### Deliverables:

1. Build `.mojopkg` package for ExTensors module
1. Installation documentation
1. API reference documentation
1. Usage examples and tutorials
1. Performance benchmarks
1. Integration guide for other ML Odyssey components

### Success criteria:

- Package builds successfully
- Examples run and produce correct output
- Documentation is complete and clear

### Phase 5: Cleanup (Issue #222) - After parallel phases

**Objective:** Refactor and finalize implementation

### Tasks:

1. Code review for consistency and style
1. Performance profiling and optimization
1. Documentation polish
1. Remove technical debt
1. Final integration testing

### Success criteria:

- Code passes all quality gates
- Performance meets benchmarks
- Documentation is comprehensive
- Ready for production use

## Expected Outputs

Please provide your deliverables in this structure:

### 1. Design Document (Phase 1 / Issue #218)

```xml
<design_document>
  <api_specification>
    <!-- Complete API with type signatures -->
  </api_specification>

  <architecture_decisions>
    <!-- Static vs Dynamic design -->
    <!-- Memory layout choices -->
    <!-- Type system approach -->
    <!-- Broadcasting strategy -->
  </architecture_decisions>

  <implementation_strategy>
    <!-- Step-by-step implementation plan -->
    <!-- Dependencies and ordering -->
    <!-- Risk mitigation -->
  </implementation_strategy>

  <performance_strategy>
    <!-- SIMD optimization points -->
    <!-- Static tensor benefits -->
    <!-- Memory access patterns -->
  </performance_strategy>
</design_document>
```text

### 2. Test Suite (Phase 2 / Issue #219)

```xml
<test_suite>
  <test_plan>
    <!-- Test categories and coverage map -->
  </test_plan>

  <test_files>
    <!-- Mojo test files with comprehensive cases -->
  </test_files>

  <benchmarks>
    <!-- Performance benchmark suite -->
  </benchmarks>
</test_suite>
```text

### 3. Implementation (Phase 3 / Issue #220)

```xml
<implementation>
  <core_types>
    <!-- ExTensor struct with DType parametrization -->
  </core_types>

  <operations>
    <!-- All operation implementations -->
  </operations>

  <utilities>
    <!-- Helper functions and internal APIs -->
  </utilities>
</implementation>
```text

### 4. Package (Phase 4 / Issue #221)

```xml
<package>
  <build_artifacts>
    <!-- .mojopkg build configuration -->
  </build_artifacts>

  <documentation>
    <!-- API reference, tutorials, examples -->
  </documentation>

  <integration>
    <!-- Integration guide and compatibility notes -->
  </integration>
</package>
```text

### 5. Cleanup Report (Phase 5 / Issue #222)

```xml
<cleanup_report>
  <refactoring>
    <!-- Code improvements and optimizations -->
  </refactoring>

  <final_benchmarks>
    <!-- Performance results -->
  </final_benchmarks>

  <lessons_learned>
    <!-- Implementation insights and future improvements -->
  </lessons_learned>
</cleanup_report>
```text

## Constraints and Guidelines

### MUST Requirements

1. ✅ Use Mojo language exclusively (no Python for core tensor implementation)
1. ✅ Implement all operations from "Complete Tensor API" section
1. ✅ Follow NumPy-style broadcasting rules exactly
1. ✅ Use IEEE 754 semantics for floating-point edge cases
1. ✅ Provide comprehensive test coverage (>95%)
1. ✅ Follow ML Odyssey coding standards (KISS, YAGNI, TDD, DRY, SOLID)
1. ✅ Memory safety via Mojo's ownership system

### SHOULD Requirements

1. ✅ Use SIMD optimization for element-wise operations
1. ✅ Provide clear, actionable error messages
1. ✅ Include docstrings for all public APIs
1. ✅ Benchmark SIMD operation performance
1. ✅ Use strided memory for zero-copy views where possible

### COULD Requirements (Future Extensions)

1. ⏸️ Lazy evaluation and operation fusion
1. ⏸️ Automatic differentiation (autograd)
1. ⏸️ GPU acceleration support
1. ⏸️ Distributed tensor operations
1. ⏸️ Complex number support
1. ⏸️ Einstein summation notation (einsum)

### MUST NOT

1. ❌ Sacrifice type safety for convenience
1. ❌ Implement implicit type coercion
1. ❌ Copy NumPy/PyTorch code (implement from scratch in Mojo)
1. ❌ Add features beyond the complete API (follow YAGNI)
1. ❌ Sacrifice memory safety for performance

## Success Criteria

### Functional Requirements

- [ ] All operations from "Complete Tensor API" implemented
- [ ] Broadcasting works for all operation types
- [ ] All data types supported correctly
- [ ] Arbitrary dimensions supported (0D to 16D)
- [ ] All tests pass (>95% coverage)
- [ ] Memory safety verified (no leaks, no use-after-free)

### Performance Requirements

- [ ] SIMD operations achieve expected vectorization speedup
- [ ] Memory layout enables cache-efficient access
- [ ] No unnecessary allocations in hot paths
- [ ] Zero-copy views for transpose, reshape, slice operations

### Quality Requirements

- [ ] Code passes `mojo format` and pre-commit hooks
- [ ] Comprehensive documentation (API reference, tutorials, examples)
- [ ] Clear, actionable error messages
- [ ] Design decisions documented with rationale
- [ ] Package builds successfully

### Integration Requirements

- [ ] Compatible with ML Odyssey architecture
- [ ] Usable from other Mojo modules
- [ ] Examples demonstrate real ML use cases
- [ ] Ready for LeNet-5 implementation (04-first-paper section)

## References and Resources

### Mojo Documentation

- [Mojo Type System](https://docs.modular.com/mojo/manual/types)
- [Mojo Structs](https://docs.modular.com/mojo/manual/basics/)
- [Mojo Parametric Types](https://docs.modular.com/mojo/manual/)
- [Mojo Traits](https://docs.modular.com/mojo/manual/)

### Array API Standard

- [Array API Standard 2024](https://data-apis.org/array-api/2024.12/)
- [NumPy Enhancement Proposal 47 (Array API)](https://numpy.org/neps/nep-0047-array-api-standard.html)

### Tensor Libraries (for API reference, not code)

- PyTorch Tensor API
- NumPy ndarray API
- JAX Array API

### Tensor Calculus

- Tensor algebra operations (addition, product, contraction)
- Ricci calculus and Einstein notation
- Covariant and contravariant indices

### ML Odyssey Documentation

- [CLAUDE.md](../../../../../../../home/user/ml-odyssey/CLAUDE.md) - Project guidelines
- [Agent Hierarchy](../../../../../../../home/user/ml-odyssey/agents/agent-hierarchy.md)
- [5-Phase Workflow](../../../../../../../home/user/ml-odyssey/notes/review/README.md)

## Questions to Address in Your Design

1. **Shape Representation:** How should shapes be represented at runtime?
   - Dynamic array/list? Fixed-size buffer with capacity? SIMD-optimized storage?
   - What's the maximum number of dimensions to support?

1. **Broadcasting Implementation:** Should broadcasting be:
   - Separate function/trait that operations call?
   - Integrated into each operation?
   - Helper utilities that compute output shapes?

1. **Memory Ownership:** What ownership model should tensor operations use?
   - Return new owned tensor (safe but allocates)?
   - Support in-place mutations via `inout` parameters?
   - Provide both owned and in-place variants?
   - How to handle borrowed tensors for read-only operations?

1. **Zero-Copy Operations:** Which operations can be zero-copy?
   - Zero-copy: transpose, reshape, slice (create views with different strides)
   - Requires allocation: arithmetic, matmul, reductions
   - How to indicate view vs copy in API? Explicit method names or flags?

1. **Trait Design:** What traits should ExTensor implement?
   - Single `Tensor` trait with all operations?
   - Separate `TensorOps`, `TensorShape`, `TensorDType` traits?
   - Standard Mojo traits (Stringable, Representable, etc.)?

1. **Error Handling:** How to provide clear error messages?
   - Custom error types or strings?
   - Validation at operation entry points?
   - Debug vs release mode validation?

1. **SIMD Vectorization:** Where and how to apply SIMD?
   - Element-wise ops (add, mul, div, etc.)
   - Reduction ops (sum, mean, max, min)
   - What SIMD width? Fixed or parametric?
   - How to handle non-aligned memory?

1. **Stride Calculation:** How to handle strided memory?
   - Row-major (C-order) vs column-major (Fortran-order)?
   - Stride storage: separate array or computed on-demand?
   - Contiguous checking: flag or computed property?

---

## How to Use This Prompt

This prompt is designed for implementation across multiple work sessions:

### For Planning (Issue #218):

Use sections: Technical Requirements, Implementation Approach (Phase 1), Questions to Address

### For Testing (Issue #219):

Use sections: Minimal Complete Tensor API, Implementation Approach (Phase 2), Success Criteria

### For Implementation (Issue #220):

Use sections: Technical Requirements, Minimal Complete Tensor API, Implementation Approach (Phase 3), Constraints

### For Packaging (Issue #221):

Use sections: Implementation Approach (Phase 4), Success Criteria (Integration Requirements)

### For Cleanup (Issue #222):

Use sections: Implementation Approach (Phase 5), Success Criteria (all), Quality Requirements

---

**Remember:** This is a foundational component for ML Odyssey. Take time to design it correctly. The static/dynamic tensor dual system is unique and powerful - make it elegant and performant. Focus on clarity, safety, and performance in that order.

Good luck! 🚀
