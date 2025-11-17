# ExTensors (Extensible Tensors)

## Overview

Design and implement ExTensors, a comprehensive tensor class that serves as the foundational data structure for all neural network operations in ML Odyssey. ExTensors supports both static (compile-time shape optimization) and dynamic (runtime flexibility) variants, arbitrary data types, arbitrary dimensions, and a complete set of tensor operations based on tensor calculus principles. The implementation follows the Python Array API Standard 2024 and provides NumPy-style broadcasting.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

This component follows the 5-phase development workflow. Each phase has its own issue:

- Issue #218: [Plan] ExTensors - Design and Documentation (this plan)
- Issue #219: [Test] ExTensors - Test-driven development
- Issue #220: [Impl] ExTensors - Implementation
- Issue #221: [Package] ExTensors - Packaging and distribution
- Issue #222: [Cleanup] ExTensors - Refactoring and finalization

Note: The original child plans (01-basic-arithmetic, 02-matrix-ops, 03-reduction-ops) are superseded by this unified ExTensors design.

## Inputs

- Mojo's type system (structs, parametric types, traits)
- Python Array API Standard 2024 specification
- Tensor calculus mathematical foundations
- Broadcasting rules from NumPy/PyTorch conventions
- SIMD optimization opportunities in Mojo

## Outputs

- **ExStaticTensor:** Compile-time optimized tensor with static shapes
- **ExTensor:** Runtime flexible tensor with dynamic shapes
- **Complete tensor API:** Creation, arithmetic, matrix ops, reductions, shape manipulation, indexing, comparisons
- **Broadcasting support:** NumPy-style broadcasting for all compatible operations
- **Multi-dtype support:** Float16/32/64, Int8/16/32/64, UInt8/16/32/64, Bool
- **Arbitrary dimensions:** Support for 0D (scalars) through N-D tensors
- **Comprehensive tests:** >95% coverage with performance benchmarks
- **Documentation:** API reference, tutorials, examples, design rationale

## Steps

1. **Plan Phase (Issue #218):**
   - Design static vs dynamic tensor architecture
   - Specify complete API surface (40+ operations)
   - Document broadcasting algorithm
   - Define memory layout and ownership model
   - Specify error handling strategy

2. **Test Phase (Issue #219):** (parallel after Plan)
   - Write comprehensive test suite following TDD
   - Create performance benchmarks
   - Test all operations with multiple dtypes
   - Verify broadcasting rules
   - Test edge cases (empty tensors, scalars, overflow)

3. **Implementation Phase (Issue #220):** (parallel after Plan)
   - Implement ExStaticTensor and ExTensor structs
   - Implement all creation operations
   - Implement arithmetic operations with broadcasting
   - Implement matrix operations
   - Implement reduction operations
   - Implement shape manipulation
   - Implement indexing and slicing
   - Apply SIMD optimizations

4. **Package Phase (Issue #221):** (parallel after Plan)
   - Build .mojopkg package
   - Create installation documentation
   - Write API reference documentation
   - Create usage examples and tutorials
   - Generate performance benchmark reports

5. **Cleanup Phase (Issue #222):** (after parallel phases)
   - Code review and refactoring
   - Performance profiling and optimization
   - Documentation polish
   - Final integration testing

## Success Criteria

- [ ] Static and dynamic tensor variants both implemented
- [ ] All 40+ operations from Array API Standard 2024 working
- [ ] Broadcasting works correctly for all compatible operations
- [ ] All data types (Float16/32/64, Int8/16/32/64, UInt8/16/32/64, Bool) supported
- [ ] Arbitrary dimensions (0D to 16D) supported
- [ ] Test coverage >95%
- [ ] Static tensors show ≥2x performance improvement vs dynamic
- [ ] SIMD optimizations applied and verified
- [ ] Comprehensive documentation (API ref, tutorials, examples)
- [ ] Package builds successfully
- [ ] All child phases (219-222) completed successfully

## Notes

**Key Design Principles:**

- **Dual type system:** ExStaticTensor (compile-time optimized) and ExTensor (runtime flexible) sharing common trait interface
- **Tensor calculus foundation:** Operations preserve mathematical semantics (rank preservation, proper broadcasting, contraction)
- **Array API Standard 2024 compliance:** Follow latest standard for ecosystem compatibility
- **YAGNI approach:** Implement minimal complete API, defer advanced features (autograd, GPU, einsum)
- **Safety first:** Leverage Mojo's type system for compile-time validation where possible

**Performance Strategy:**

- Static tensors enable compile-time shape checking and SIMD vectorization
- Row-major (C-order) memory layout for cache efficiency
- Strided memory for zero-copy slicing
- Operation fusion opportunities (future optimization)

**References:**

- [ExTensors Implementation Prompt](/home/user/ml-odyssey/notes/issues/218/extensor-implementation-prompt.md) - Comprehensive design specification
- [Array API Standard 2024](https://data-apis.org/array-api/2024.12/) - API reference
- [Mojo Type System](https://docs.modular.com/mojo/manual/types) - Language reference
