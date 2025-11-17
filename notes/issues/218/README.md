# Issue #218: [Plan] ExTensors - Design and Documentation

## Objective

Design and document the architecture for ExTensors (Extensible Tensors), a comprehensive tensor class that serves as the foundational data structure for all neural network operations in ML Odyssey. ExTensors provides both static (compile-time optimized) and dynamic (runtime flexible) variants, supports arbitrary data types and dimensions, and implements a complete tensor API based on the Array API Standard 2024 and tensor calculus principles.

## Deliverables

- **ExStaticTensor specification** - Compile-time optimized tensor with static shapes
- **ExTensor specification** - Runtime flexible tensor with dynamic shapes
- **Complete API specification** - 40+ operations including:
  - Creation operations (zeros, ones, full, arange, from_array)
  - Element-wise arithmetic (add, subtract, multiply, divide, power, negative)
  - Matrix operations (matmul, transpose, dot)
  - Reduction operations (sum, mean, max, min, count_nonzero, cumulative_sum/prod)
  - Shape manipulation (reshape, squeeze, expand_dims, flatten, concatenate, stack)
  - Indexing and slicing (getitem, setitem, take_along_axis)
  - Comparison operations (equal, greater, less, etc.)
  - Utility operations (copy, diff, clip, abs, sqrt)
- **Broadcasting specification** - NumPy-style broadcasting rules for all compatible operations
- **Multi-dtype support** - Float16/32/64, Int8/16/32/64, UInt8/16/32/64, Bool
- **Memory layout specification** - Row-major (C-order), strided memory, ownership model
- **Performance strategy** - SIMD optimization, static tensor benefits, cache efficiency
- **Comprehensive design documentation** - Architecture decisions, API contracts, edge case handling, rationale

## Success Criteria

- [ ] Static vs Dynamic tensor architecture clearly specified
- [ ] All 40+ operations from Array API Standard 2024 specified
- [ ] Broadcasting algorithm documented with examples
- [ ] Memory layout and ownership model defined
- [ ] Error handling strategy specified (compile-time vs runtime)
- [ ] Multi-dtype support matrix documented
- [ ] Arbitrary dimension support (0D to N-D) specified
- [ ] Performance optimization strategy documented
- [ ] SIMD vectorization opportunities identified
- [ ] Edge case handling specified (empty tensors, scalars, overflow, NaN, inf)
- [ ] Design documentation is comprehensive and unambiguous
- [ ] Specifications ready for Test, Implementation, and Package phases

## Design Decisions

### 1. Dual Type System (Static + Dynamic)

**Decision**: Implement both ExStaticTensor and ExTensor variants sharing a common trait interface.

**Rationale**:

- Static tensors enable compile-time shape validation and SIMD optimization
- Dynamic tensors provide runtime flexibility for research and experimentation
- Common trait ensures API compatibility between variants
- Mojo's parametric types allow compile-time shape specification without runtime overhead
- Users can opt into performance when shapes are known, default to flexibility otherwise

**Trade-offs**:

- More implementation complexity vs single dynamic-only approach
- Benefit: 2-10x performance improvement for static paths in tight loops
- Cost: Maintaining two code paths (mitigated by shared trait)

### 2. Tensor Calculus Foundation

**Decision**: Ground all operations in tensor calculus principles (tensor algebra).

**Rationale**:

- Ensures mathematical correctness and consistency
- Tensor addition/subtraction only for same-rank tensors (preserves structure)
- Tensor product (outer product) increases rank appropriately
- Contraction (Einstein summation) decreases rank
- Clear semantics prevent subtle mathematical errors

**Examples**:

- Element-wise ops preserve shape: `(3,4,5) + (3,4,5) → (3,4,5)`
- Outer product increases rank: `(3,4) ⊗ (5,6) → (3,4,5,6)`
- Matmul contracts inner dims: `(3,4) @ (4,5) → (3,5)`

### 3. Array API Standard 2024 Compliance

**Decision**: Follow Python Array API Standard 2024 for API design.

**Rationale**:

- Ecosystem compatibility (NumPy, PyTorch, JAX conventions)
- Well-tested API surface (proven in production)
- Future interoperability with Python-based tools
- Includes latest operations (count_nonzero, cumulative_prod, take_along_axis, diff)

**Coverage**: 40+ operations across 7 categories (creation, arithmetic, matrix, reduction, shape, indexing, comparison)

### 4. Broadcasting Strategy

**Decision**: Implement NumPy-style broadcasting for all compatible operations.

**Algorithm**:

1. Compare shapes element-wise from right to left
2. Dimensions are compatible if equal or one is 1
3. Missing dimensions treated as 1
4. Output shape is element-wise maximum of input shapes

**Rationale**:

- De facto standard in ML (NumPy, PyTorch, JAX)
- Efficient computation without data replication
- Supports common patterns (batch ops, bias addition, normalization)

**Examples**:

```text
(3, 4, 5) + (4, 5)    → (3, 4, 5)  # Missing dim = 1
(3, 1, 5) + (3, 4, 5) → (3, 4, 5)  # Size 1 broadcasts
(3, 4, 5) + (3, 4, 1) → (3, 4, 5)  # Size 1 broadcasts
```

### 5. Multi-Dtype Support

**Decision**: Support all common numeric types via Mojo's DType system.

**Supported types**:

- Float: DType.float16, float32, float64
- Integer: DType.int8, int16, int32, int64
- Unsigned: DType.uint8, uint16, uint32, uint64
- Boolean: DType.bool

**Rationale**:

- Covers all ML use cases (float32 most common, int8/uint8 for quantization)
- Parametric DType enables type-safe generic code
- Explicit type conversion prevents implicit coercion bugs
- Future extension: bfloat16, complex types

### 6. Memory Layout and Ownership

**Decision**: Row-major (C-order) strided memory layout with explicit ownership.

**Layout**:

- Row-major (C-order) as default (cache-friendly for most ML ops)
- Strided memory for zero-copy slicing and transpose
- Contiguous memory guarantee for SIMD operations

**Ownership**:

- Tensors own their memory by default
- `borrowed` for read-only access (no allocation)
- `inout` for in-place mutations
- Explicit copy operation when needed

**Rationale**:

- Row-major matches NumPy/PyTorch defaults
- Strided memory enables efficient views
- Mojo's ownership system prevents memory leaks and use-after-free

### 7. Error Handling Strategy

**Decision**: Compile-time validation for static tensors, runtime validation for dynamic tensors.

**Static tensors (ExStaticTensor)**:

- Shape mismatches caught at compile-time
- Type mismatches caught at compile-time
- Zero runtime overhead for validation

**Dynamic tensors (ExTensor)**:

- Shape validation at runtime with clear error messages
- Type validation at runtime
- Graceful error messages (e.g., "Cannot broadcast (3,4,5) with (2,4,5)")

**Floating-point edge cases** (IEEE 754):

- `x / 0.0` where `x > 0` → `+inf`
- `x / 0.0` where `x < 0` → `-inf`
- `0.0 / 0.0` → `NaN`
- Overflow → saturate to infinity

**Rationale**:

- Leverages Mojo's type system for safety
- Static path gets performance and safety
- Dynamic path gets flexibility with good error messages
- IEEE 754 enables gradient computation (NaN propagation)

### 8. API Design

**Decision**: Provide both function and operator interfaces.

**Function API**: `add(a, b)`, `matmul(a, b)`, `sum(a, axis=0)`

**Operator API**: `a + b`, `a @ b`, `a[0:10, :]`

**Rationale**:

- Function API: explicit, unambiguous, supports keyword args
- Operator API: natural mathematical syntax, familiar to ML practitioners
- Both patterns common in tensor libraries (PyTorch, NumPy)
- Mojo supports operator overloading

### 9. YAGNI Scope Limitation

**Decision**: Implement minimal complete API, defer advanced features.

**In scope** (minimal complete set):

- 40+ core operations from Array API Standard 2024
- Static and dynamic variants
- Broadcasting
- Multi-dtype support
- SIMD optimization

**Out of scope** (future extensions):

- Automatic differentiation (autograd)
- GPU acceleration
- Distributed tensors
- Complex numbers
- Einstein summation (einsum)
- Lazy evaluation and operation fusion

**Rationale**:

- YAGNI: Don't implement until needed
- Focus on solid foundation first
- Autograd planned for separate component (02-shared-library/02-training-utils)
- GPU planned for future optimization phase

### 10. Performance Optimization Strategy

**Decision**: Focus on SIMD vectorization and static tensor compile-time optimization.

**Optimizations**:

- SIMD vectorization for element-wise operations
- Compile-time shape optimization for static tensors
- Row-major memory layout for cache efficiency
- Strided memory for zero-copy views
- Loop unrolling for small fixed-size tensors

**Benchmarks**:

- Target: ≥2x speedup for static vs dynamic in tight loops
- Verify SIMD speedup for element-wise ops
- Profile memory access patterns

**Rationale**:

- SIMD provides immediate performance benefit
- Static tensors unlock compile-time optimization
- Row-major layout matches CPU cache lines
- Measurable performance targets ensure optimization effort is effective

## References

### Source Plan

- [ExTensors Plan](/home/user/ml-odyssey/notes/plan/02-shared-library/01-core-operations/01-tensor-ops/plan.md)

### Implementation Prompt

- [ExTensors Implementation Prompt](/home/user/ml-odyssey/notes/issues/218/extensor-implementation-prompt.md) - Comprehensive design specification following Anthropic's prompting best practices

### Related Issues (5-Phase Workflow)

- Issue #218: [Plan] ExTensors - Design and Documentation (this issue)
- Issue #219: [Test] ExTensors - Test-driven development (parallel phase)
- Issue #220: [Impl] ExTensors - Implementation (parallel phase)
- Issue #221: [Package] ExTensors - Packaging and distribution (parallel phase)
- Issue #222: [Cleanup] ExTensors - Refactoring and finalization (sequential phase)

### External Standards and Documentation

- [Array API Standard 2024](https://data-apis.org/array-api/2024.12/) - API specification reference
- [Mojo Type System](https://docs.modular.com/mojo/manual/types) - Language reference for structs, parametric types, traits
- [NumPy Broadcasting Rules](https://numpy.org/doc/stable/user/basics.broadcasting.html) - Broadcasting semantics

### ML Odyssey Documentation

- [Agent Hierarchy](/home/user/ml-odyssey/agents/agent-hierarchy.md)
- [5-Phase Workflow](/home/user/ml-odyssey/notes/review/README.md)
- [CLAUDE.md](/home/user/ml-odyssey/CLAUDE.md) - Project guidelines

## Key Questions to Address in Design

These questions must be answered during the planning phase:

1. **Static vs Dynamic API Unification**: Should there be a unified API that automatically selects ExStaticTensor vs ExTensor based on compile-time information, or should users explicitly choose?

2. **Shape Representation**: How should shapes be represented?
   - Static: Parametric types? Variadic parameters? Fixed-size array?
   - Dynamic: Runtime array? List? SIMD-optimized buffer?

3. **Trait Design**: What trait(s) should both variants implement?
   - Common `Tensor` trait with all operations?
   - Separate `TensorOps`, `TensorShape`, `TensorDType` traits?
   - How to handle static-only optimizations in trait?

4. **Broadcasting Implementation**: Should broadcasting be:
   - Separate function/trait that operations call?
   - Integrated into each operation?
   - Compile-time for static tensors, runtime for dynamic?

5. **Memory Ownership in Operations**: What ownership model for tensor operations?
   - Return new owned tensor (safe but allocates)?
   - Support in-place mutations via `inout` parameters?
   - Provide both owned and in-place variants?

6. **Zero-Copy Operations**: Which operations can be zero-copy?
   - Zero-copy: transpose, reshape, slice (views)
   - Requires allocation: arithmetic, matmul, reductions
   - How to indicate view vs copy in API?

7. **Testing Strategy**: How to avoid duplicating tests for static and dynamic?
   - Parametric test functions over both types?
   - Shared test cases with type-specific setup?
   - Separate test files or unified?

8. **SIMD Vectorization**: Where to apply SIMD?
   - Element-wise ops (add, mul, div, etc.)
   - Reduction ops (sum, mean, max, min)
   - What SIMD width? Parametric over width?

## Implementation Notes

This section will be updated as the planning phase progresses with any discoveries, clarifications, or decisions made during design discussions.

### Design Session 1: Initial Scope Definition

- **Date**: 2025-11-17
- **Scope change**: Reinterpreted issues #218-222 from operation-based (basic-arithmetic, matrix-ops, reduction-ops) to phase-based (Plan, Test, Impl, Package, Cleanup) for comprehensive ExTensors implementation
- **Key decision**: ExTensors will be the foundational tensor class, not just a collection of operations
- **Research completed**: Array API Standard 2024, Mojo type system, tensor calculus foundations, Anthropic prompting best practices
- **Artifacts created**: `extensor-implementation-prompt.md` - comprehensive design specification

---

**Status**: Planning phase - Initial design document created

**Next Steps**:

1. Answer the 8 key design questions above
2. Create detailed API specification document (function signatures, type annotations)
3. Document broadcasting algorithm with pseudocode
4. Specify memory layout and stride calculations
5. Define trait interfaces for ExStaticTensor and ExTensor
6. Document SIMD optimization opportunities
7. Create examples showing static vs dynamic usage patterns
8. Prepare specifications for Test phase (issue #219)
9. Prepare specifications for Implementation phase (issue #220)
10. Prepare specifications for Package phase (issue #221)
