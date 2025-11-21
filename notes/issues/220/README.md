# Issue #220: [Implementation] ExTensors - Core Implementation

## Objective

Implement the ExTensor struct and all 150+ operations following Test-Driven Development (TDD) principles. This implementation will serve as the foundational data structure for all neural network operations in ML Odyssey.

## Deliverables

### Source Files

- `src/extensor/extensor.mojo` - Core ExTensor struct definition
- `src/extensor/creation.mojo` - Creation operations (zeros, ones, full, etc.)
- `src/extensor/arithmetic.mojo` - Arithmetic operations with broadcasting
- `src/extensor/bitwise.mojo` - Bitwise operations for integer/bool tensors
- `src/extensor/comparison.mojo` - Comparison operations
- `src/extensor/pointwise_math.mojo` - Pointwise math operations (trig, exp, log, etc.)
- `src/extensor/matrix.mojo` - Matrix operations (matmul, transpose, etc.)
- `src/extensor/reduction.mojo` - Reduction operations (sum, mean, max, etc.)
- `src/extensor/shape.mojo` - Shape manipulation operations
- `src/extensor/indexing.mojo` - Indexing and slicing operations
- `src/extensor/utility.mojo` - Utility and inspection operations
- `src/extensor/broadcasting.mojo` - Broadcasting infrastructure
- `src/extensor/__init__.mojo` - Package initialization

### Documentation

- API documentation (docstrings in code)
- Implementation notes (this file)
- Design decisions and rationale

## Implementation Strategy

### Phase 1: Core Infrastructure

1. **ExTensor Struct Definition**
   - Dynamic shape representation (runtime-determined)
   - DType parametrization for arbitrary data types
   - Memory layout (row-major, strided)
   - Ownership model (owned, borrowed, inout)

1. **Memory Management**
   - Buffer allocation and deallocation
   - Strided memory layout for zero-copy views
   - Contiguous memory guarantee for SIMD operations
   - Reference counting or ownership transfer

1. **Shape and Stride Utilities**
   - Shape validation
   - Stride calculation for row-major layout
   - Broadcast shape computation
   - Contiguity checking

### Phase 2: TDD Implementation Cycle

For each operation category (following test file order):

1. **Red** - Run tests and see them fail
1. **Green** - Implement minimal code to make tests pass
1. **Refactor** - Clean up implementation while keeping tests green

**Implementation Order** (matches test order):

1. Creation operations (foundation for all tests)
1. Arithmetic operations (core functionality)
1. Broadcasting (enables arithmetic with different shapes)
1. Comparison operations (needed for testing)
1. Shape manipulation
1. Reduction operations
1. Matrix operations
1. Pointwise math operations
1. Bitwise operations
1. Indexing and slicing
1. Utility operations
1. Edge case handling
1. Memory safety verification
1. Performance optimization (SIMD)

### Phase 3: Optimization

1. **SIMD Vectorization**
   - Element-wise operations (add, mul, div, etc.)
   - Reduction operations (sum, mean, max, min)
   - Appropriate SIMD width selection

1. **Memory Optimizations**
   - Zero-copy views (transpose, reshape, slice)
   - Cache-friendly memory access patterns
   - Minimize allocations in hot paths

1. **Broadcasting Optimizations**
   - Efficient broadcasting without materialization
   - Specialized paths for common patterns

## Success Criteria

- [ ] All tests from Issue #219 pass
- [ ] Test coverage >95%
- [ ] All 150+ operations implemented correctly
- [ ] Broadcasting works for all operation types
- [ ] All data types supported (float16/32/64, int8/16/32/64, uint8/16/32/64, bool)
- [ ] Arbitrary dimensions supported (0D to 16D)
- [ ] Memory safety verified (no leaks, no use-after-free)
- [ ] SIMD optimizations demonstrate expected speedup
- [ ] Code passes `mojo format` and pre-commit hooks
- [ ] Clear, actionable error messages for dimension mismatches
- [ ] Zero-copy operations work correctly (transpose, reshape, slice)

## Technical Design Decisions

### 1. Dynamic-Only Design

**Decision:** Implement only dynamic tensors (ExTensor) initially.

### Rationale:

- Simpler implementation following KISS principle
- Sufficient for research and experimentation
- Can add static tensors later if performance profiling shows need
- Dynamic tensors are more flexible for prototyping

### 2. Shape Representation

**Decision:** Use `DynamicVector[Int]` or similar for runtime shape storage.

### Rationale:

- Allows arbitrary number of dimensions
- Efficient for small dimension counts (typical: 2-5)
- Can be optimized with fixed-size buffer if needed

### 3. Broadcasting Implementation

**Decision:** Separate broadcasting utility that computes output shapes and validates compatibility.

### Rationale:

- Reusable across all operations
- Clear separation of concerns
- Easy to test broadcasting rules independently
- Follows DRY principle

### 4. Memory Ownership Model

**Decision:** Provide both owned and in-place variants:

- Arithmetic operations return new tensors by default
- In-place variants via dunder methods (`__iadd__`, etc.)
- Read-only operations accept borrowed tensors
- Mutations require `inout` parameters

### Rationale:

- Safe by default (owned tensors prevent accidental aliasing)
- Performance optimization available (in-place when needed)
- Clear API semantics

### 5. Zero-Copy Operations

**Decision:** Transpose, reshape, and slice create views (zero-copy) when possible.

### Rationale:

- Significant performance improvement
- NumPy/PyTorch users expect this behavior
- Strided memory layout enables views

**Implementation:** Use flags or separate methods to indicate view vs copy.

### 6. Trait Design

**Decision:** Implement standard Mojo traits (Stringable, Representable, Sized, etc.) as appropriate.

### Rationale:

- Interoperability with Mojo ecosystem
- Standard protocols for common operations
- Clear API contracts

### 7. Error Handling

**Decision:** Runtime validation with descriptive error messages using Mojo's `raises` mechanism.

### Rationale:

- Clear error messages improve developer experience
- Runtime validation catches shape mismatches early
- Mojo's error handling is lightweight

### 8. SIMD Vectorization

**Decision:** Apply SIMD to element-wise operations with configurable width.

### Rationale:

- Significant performance improvement for large tensors
- Mojo's SIMD support makes this straightforward
- Benchmark to verify speedup

## References

- [ExTensors Implementation Prompt](../../../../../../../home/user/ml-odyssey/notes/issues/218/extensor-implementation-prompt.md)
- [Array API Standard 2024](https://data-apis.org/array-api/2024.12/)
- [Mojo Type System](https://docs.modular.com/mojo/manual/types)
- [Issue #219: Test Specification](../../../../../../../home/user/ml-odyssey/notes/issues/219/README.md)

## Implementation Notes

### Implementation Session 1

- **Date:** 2025-11-17
- **Status:** Starting implementation
- **Approach:** TDD - write tests first, then implement
- **First milestone:** Core struct + creation operations

---

**Status:** Ready to begin implementation

### Next Steps:

1. Create minimal ExTensor struct (just enough to compile)
1. Run creation operation tests (they will fail)
1. Implement creation operations to make tests pass
1. Continue with arithmetic operations following TDD cycle
