# Matrix Ops

## Overview
Implement matrix operations essential for linear algebra computations in neural networks. This includes matrix multiplication (matmul) for layer computations and transpose for reshaping and gradient calculations. These operations are core to all neural network architectures.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- Tensors with 2D or higher dimensions
- Matrix multiplication dimension compatibility rules
- Transpose axis specifications

## Outputs
- Matrix multiplication (matmul) operation
- Transpose operation with flexible axis ordering
- Support for batched operations

## Steps
1. Implement matrix multiplication with dimension checking
2. Create transpose operation with configurable axes
3. Add support for batched matrix operations
4. Ensure numerical stability for large matrices

## Success Criteria
- [ ] Matmul produces mathematically correct results
- [ ] Transpose correctly reorders tensor dimensions
- [ ] Operations handle dimension mismatches appropriately
- [ ] Batched operations work correctly

## Notes
Start with simple 2D matrix multiplication, then extend to batched operations. Ensure dimension checking provides clear error messages. Use straightforward implementations without low-level optimizations.
