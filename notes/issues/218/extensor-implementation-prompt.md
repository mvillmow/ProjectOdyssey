# ExTensors Implementation Prompt

## Context and Motivation

You are tasked with designing and implementing **ExTensors** (Extensible Tensors), a comprehensive tensor class for the ML Odyssey project. This implementation will serve as the foundational data structure for all neural network operations in the codebase. ExTensors must be both flexible enough to handle dynamic workloads and optimized enough to achieve maximum performance when tensor shapes are known at compile-time.

**Why this matters:**

- Tensors are the fundamental data structure in all deep learning frameworks
- Performance-critical ML workloads require compile-time optimization when possible
- Research implementations need flexibility to experiment with arbitrary tensor shapes
- Mojo's unique type system allows us to unify both dynamic and static tensor paths in a single elegant design

## Task Overview

Re-interpret GitHub issues #218-222 (originally scoped for basic arithmetic operations) to instead implement a complete, production-ready tensor class called **ExTensors**.

**Original scope:** Basic arithmetic operations (add, subtract, multiply, divide)
**New scope:** Complete tensor infrastructure with static/dynamic variants

## Technical Requirements

### 1. Dual Type System (Static + Dynamic)

Implement both static and dynamic tensor variants using Mojo's parametric type system:

**Static Tensor (ExTensorStatic):**
- Shape known at compile-time via parametric types (e.g., `ExTensorStatic[DType.float32, 3, 224, 224]`)
- Optimized code paths with compile-time bounds checking
- Zero runtime overhead for shape validation
- SIMD vectorization opportunities
- Memory layout determined at compile-time

**Dynamic Tensor (ExTensorDynamic):**
- Shape determined at runtime (default variant)
- Flexible for research and experimentation
- Runtime shape validation
- Support for dynamic reshaping and shape inference
- Compatible with Python-style workflow

**Relationship between variants:**
- Both should share a common trait interface (e.g., `TensorOps` trait)
- ExTensorDynamic is the default for ease of use
- ExTensorStatic provides opt-in optimization when shapes are known
- Seamless conversion between static and dynamic when needed

### 2. Tensor Calculus Foundation

Implement operations grounded in tensor calculus principles:

**Core tensor algebra operations:**
- **Tensor addition/subtraction:** Only defined for tensors of same rank and compatible dimensions
- **Tensor product (outer product):** Combines tensors of rank (m, n) to produce rank (m+n) tensor
- **Contraction:** Reduces tensor rank by summing over matched indices (Einstein summation)
- **Scalar multiplication:** Scales all tensor elements
- **Element-wise operations:** Apply functions element-wise while preserving tensor structure

**Type preservation:**
- Operations should preserve or predictably transform tensor rank
- Clear semantics for covariant vs contravariant indices (future extension)
- Support for Einstein summation notation (future extension)

### 3. Minimal Complete Tensor API

Based on the **Array API Standard 2024** and PyTorch/NumPy conventions, implement this minimal set of operations for a complete tensor implementation:

#### Creation Operations
- `zeros(shape, dtype)` - Create tensor filled with zeros
- `ones(shape, dtype)` - Create tensor filled with ones
- `full(shape, fill_value, dtype)` - Create tensor filled with a value
- `arange(start, stop, step, dtype)` - Create 1D tensor with evenly spaced values
- `from_array(data)` - Create tensor from array/list data

#### Element-wise Arithmetic (broadcasting support)
- `add(a, b)` / `a + b` - Element-wise addition
- `subtract(a, b)` / `a - b` - Element-wise subtraction
- `multiply(a, b)` / `a * b` - Element-wise multiplication
- `divide(a, b)` / `a / b` - Element-wise division (IEEE 754 semantics)
- `power(a, b)` / `a ** b` - Element-wise exponentiation
- `negative(a)` / `-a` - Element-wise negation

#### Matrix Operations
- `matmul(a, b)` / `a @ b` - Matrix multiplication
- `transpose(a, axes=None)` - Transpose tensor along specified axes
- `dot(a, b)` - Dot product (1D) or matrix product

#### Reduction Operations
- `sum(a, axis=None, keepdims=False)` - Sum along axis
- `mean(a, axis=None, keepdims=False)` - Mean along axis
- `max(a, axis=None, keepdims=False)` - Maximum along axis
- `min(a, axis=None, keepdims=False)` - Minimum along axis
- `count_nonzero(a, axis=None)` - Count non-zero elements (Array API 2024)
- `cumulative_sum(a, axis)` - Cumulative sum along axis
- `cumulative_prod(a, axis)` - Cumulative product (Array API 2024)

#### Shape Manipulation
- `reshape(a, new_shape)` - Reshape tensor
- `squeeze(a, axis=None)` - Remove dimensions of size 1
- `expand_dims(a, axis)` - Add dimension of size 1
- `flatten(a)` - Flatten to 1D
- `concatenate(tensors, axis)` - Join tensors along axis
- `stack(tensors, axis)` - Stack tensors along new axis

#### Indexing and Slicing
- `__getitem__(indices)` / `a[i, j, k]` - Tensor indexing
- `__setitem__(indices, value)` / `a[i, j, k] = value` - Tensor assignment
- `take_along_axis(a, indices, axis)` - Gather along axis (Array API 2024)
- `slice(a, start, end, step)` - Slice along dimensions

#### Comparison Operations
- `equal(a, b)` / `a == b` - Element-wise equality
- `greater(a, b)` / `a > b` - Element-wise greater-than
- `less(a, b)` / `a < b` - Element-wise less-than
- `greater_equal(a, b)` / `a >= b`
- `less_equal(a, b)` / `a <= b`

#### Utility Operations
- `copy(a)` - Create deep copy
- `diff(a, axis)` - Discrete difference (Array API 2024)
- `clip(a, min, max)` - Clamp values to range
- `abs(a)` - Absolute value
- `sqrt(a)` - Square root

### 4. Data Type Support

Support for arbitrary data types via Mojo's DType system:

**Initial supported types:**
- Float: `DType.float16`, `DType.float32`, `DType.float64`
- Integer: `DType.int8`, `DType.int16`, `DType.int32`, `DType.int64`
- Unsigned: `DType.uint8`, `DType.uint16`, `DType.uint32`, `DType.uint64`
- Boolean: `DType.bool`

**Extension path:**
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
2. Dimensions are compatible if they are equal or one is 1
3. Missing dimensions treated as 1
4. Output shape is element-wise maximum of input shapes

**Examples:**
```text
(3, 4, 5) + (4, 5)    → (3, 4, 5)  # Missing dimension = 1
(3, 1, 5) + (3, 4, 5) → (3, 4, 5)  # Size 1 broadcasts
(3, 4, 5) + (3, 4, 1) → (3, 4, 5)  # Size 1 broadcasts
```

### 7. Memory and Performance

**Memory layout:**
- Row-major (C-order) as default
- Strided memory layout for efficient slicing
- Contiguous memory guarantee for SIMD operations

**Performance optimizations:**
- SIMD vectorization for element-wise operations
- Compile-time shape optimization for static tensors
- Cache-friendly memory access patterns
- Lazy evaluation for operation fusion (future)

### 8. Error Handling and Safety

**Mojo's safety-first approach:**
- Compile-time shape validation for static tensors
- Runtime shape validation for dynamic tensors
- Clear error messages for dimension mismatches
- IEEE 754 semantics for floating-point edge cases:
  - `x / 0.0` where `x > 0` → `+inf`
  - `x / 0.0` where `x < 0` → `-inf`
  - `0.0 / 0.0` → `NaN`

**Edge case handling:**
- Empty tensors (0-element shapes)
- Large value overflow → saturate to infinity
- Zero-dimensional scalar tensors
- Non-broadcastable shape errors

## Implementation Approach

Follow the 5-phase development workflow:

### Phase 1: Plan (Issue #218)
**Objective:** Design ExTensors architecture and API specification

**Deliverables:**
1. Complete API specification document
2. Static vs Dynamic design decisions with rationale
3. Memory layout specification
4. Broadcasting algorithm specification
5. Data type support matrix
6. Performance optimization strategy
7. Error handling specification

**Success criteria:**
- Unambiguous API contracts for all operations
- Clear distinction between static and dynamic paths
- Comprehensive edge case documentation
- Design ready for Test and Implementation phases

### Phase 2: Test (Issue #219) - Parallel after Plan
**Objective:** Write comprehensive test suite following TDD principles

**Test categories:**
1. **Creation tests:** All creation operations with various dtypes and shapes
2. **Arithmetic tests:** Element-wise ops with broadcasting, edge cases
3. **Matrix operation tests:** Matmul, transpose with various dimensions
4. **Reduction tests:** All reductions with keepdims, axis variations
5. **Shape manipulation tests:** Reshape, squeeze, expand, concatenate
6. **Indexing tests:** Slicing, advanced indexing, assignment
7. **Broadcasting tests:** All broadcasting rule combinations
8. **Static vs Dynamic tests:** Verify optimization paths
9. **Edge case tests:** Empty tensors, scalars, overflow, NaN, inf
10. **Type tests:** All supported dtypes

**Success criteria:**
- 100% coverage of API surface
- Tests pass for both static and dynamic variants
- Performance benchmarks for static optimization benefit

### Phase 3: Implementation (Issue #220) - Parallel after Plan
**Objective:** Implement ExTensors following TDD

**Implementation order:**
1. Core tensor struct definitions (static and dynamic)
2. Memory allocation and layout
3. Creation operations
4. Basic arithmetic (no broadcasting)
5. Broadcasting infrastructure
6. Arithmetic with broadcasting
7. Matrix operations
8. Reduction operations
9. Shape manipulation
10. Indexing and slicing
11. Comparison operations
12. Utility operations

**Success criteria:**
- All tests pass
- Static tensors show measurable performance improvement
- Memory safety verified
- SIMD optimizations applied where possible

### Phase 4: Package (Issue #221) - Parallel after Plan
**Objective:** Package ExTensors for distribution and reuse

**Deliverables:**
1. Build `.mojopkg` package for ExTensors module
2. Installation documentation
3. API reference documentation
4. Usage examples and tutorials
5. Performance benchmarks
6. Integration guide for other ML Odyssey components

**Success criteria:**
- Package builds successfully
- Examples run and produce correct output
- Documentation is complete and clear

### Phase 5: Cleanup (Issue #222) - After parallel phases
**Objective:** Refactor and finalize implementation

**Tasks:**
1. Code review for consistency and style
2. Performance profiling and optimization
3. Documentation polish
4. Remove technical debt
5. Final integration testing

**Success criteria:**
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
```

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
```

### 3. Implementation (Phase 3 / Issue #220)

```xml
<implementation>
  <core_types>
    <!-- ExTensorStatic and ExTensorDynamic structs -->
  </core_types>

  <operations>
    <!-- All operation implementations -->
  </operations>

  <utilities>
    <!-- Helper functions and internal APIs -->
  </utilities>
</implementation>
```

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
```

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
```

## Constraints and Guidelines

### MUST Requirements
1. ✅ Use Mojo language exclusively (no Python for core tensor implementation)
2. ✅ Support both static and dynamic tensor variants
3. ✅ Implement all operations from "Minimal Complete Tensor API" section
4. ✅ Follow NumPy-style broadcasting rules exactly
5. ✅ Use IEEE 754 semantics for floating-point edge cases
6. ✅ Provide comprehensive test coverage (>95%)
7. ✅ Follow ML Odyssey coding standards (KISS, YAGNI, TDD, DRY, SOLID)

### SHOULD Requirements
1. ✅ Use SIMD optimization for element-wise operations
2. ✅ Optimize static tensor paths at compile-time
3. ✅ Provide clear, actionable error messages
4. ✅ Include docstrings for all public APIs
5. ✅ Benchmark static vs dynamic performance differences

### COULD Requirements (Future Extensions)
1. ⏸️ Lazy evaluation and operation fusion
2. ⏸️ Automatic differentiation (autograd)
3. ⏸️ GPU acceleration support
4. ⏸️ Distributed tensor operations
5. ⏸️ Complex number support
6. ⏸️ Einstein summation notation (einsum)

### MUST NOT
1. ❌ Sacrifice type safety for convenience
2. ❌ Implement implicit type coercion
3. ❌ Use dynamic dispatch for static tensor operations
4. ❌ Copy NumPy/PyTorch code (implement from scratch in Mojo)
5. ❌ Add features beyond the minimal API (follow YAGNI)

## Success Criteria

### Functional Requirements
- [ ] All operations from "Minimal Complete Tensor API" implemented
- [ ] Both static and dynamic variants work correctly
- [ ] Broadcasting works for all operation types
- [ ] All data types supported correctly
- [ ] Arbitrary dimensions supported (0D to 16D)
- [ ] All tests pass (>95% coverage)

### Performance Requirements
- [ ] Static tensors show ≥2x speedup vs dynamic for fixed-shape workloads
- [ ] SIMD operations achieve expected vectorization speedup
- [ ] Memory layout enables cache-efficient access
- [ ] No unnecessary allocations in hot paths

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
- [CLAUDE.md](/home/user/ml-odyssey/CLAUDE.md) - Project guidelines
- [Agent Hierarchy](/home/user/ml-odyssey/agents/agent-hierarchy.md)
- [5-Phase Workflow](/home/user/ml-odyssey/notes/review/README.md)

## Questions to Address in Your Design

1. **Static vs Dynamic API:** Should users explicitly choose `ExTensorStatic` vs `ExTensorDynamic`, or should there be a unified API that automatically selects the variant based on compile-time information?

2. **Shape Representation:** How should shapes be represented for static tensors (parametric types? variadic parameters?) vs dynamic tensors (runtime array/list?)?

3. **Broadcasting Implementation:** Should broadcasting be implemented as a separate function/trait, or integrated into each operation?

4. **Memory Ownership:** What ownership model (owned, borrowed, inout) should tensor operations use? How does this affect API ergonomics?

5. **Zero-Copy Operations:** Which operations can be zero-copy (transpose, reshape, slice) vs require allocation (arithmetic, matmul, reductions)?

6. **Trait Design:** What traits should static and dynamic tensors implement? Should there be a common `Tensor` trait?

7. **Error Strategy:** Compile-time errors for static tensors vs runtime errors for dynamic tensors - how to make this ergonomic?

8. **Testing Strategy:** How to avoid duplicating tests for static and dynamic variants? Parametric testing approach?

---

## How to Use This Prompt

This prompt is designed for implementation across multiple work sessions:

**For Planning (Issue #218):**
Use sections: Technical Requirements, Implementation Approach (Phase 1), Questions to Address

**For Testing (Issue #219):**
Use sections: Minimal Complete Tensor API, Implementation Approach (Phase 2), Success Criteria

**For Implementation (Issue #220):**
Use sections: Technical Requirements, Minimal Complete Tensor API, Implementation Approach (Phase 3), Constraints

**For Packaging (Issue #221):**
Use sections: Implementation Approach (Phase 4), Success Criteria (Integration Requirements)

**For Cleanup (Issue #222):**
Use sections: Implementation Approach (Phase 5), Success Criteria (all), Quality Requirements

---

**Remember:** This is a foundational component for ML Odyssey. Take time to design it correctly. The static/dynamic tensor dual system is unique and powerful - make it elegant and performant. Focus on clarity, safety, and performance in that order.

Good luck! 🚀
