# Issue #1975: Fix TensorDataset import and Tensor type usage

## Objective

Fix import name mismatch and undefined Tensor type usage in dataset tests.

## Deliverables

- Fixed `shared/data/datasets.mojo` with `TensorDataset` type alias
- Fixed `tests/shared/data/datasets/test_tensor_dataset.mojo` to use proper `ExTensor` type
- All tests now compile and run correctly

## Changes Made

### 1. Added TensorDataset Type Alias

**File**: `shared/data/datasets.mojo`

Added a type alias for backwards compatibility:

```mojo
# Type alias for backwards compatibility
alias TensorDataset = ExTensorDataset
```

This allows tests to import `TensorDataset` while the implementation is named `ExTensorDataset`.

### 2. Fixed Test File Tensor Type Usage

**File**: `tests/shared/data/datasets/test_tensor_dataset.mojo`

Replaced all instances of undefined `Tensor([...])` with proper `ExTensor` initialization from `List`:

**Pattern Changed**:
- From: `var data = Tensor([Float32(1.0), Float32(2.0)])`
- To: `var data_list = List[Float32](Float32(1.0), Float32(2.0))` followed by `var data = ExTensor(data_list^)`

**Functions Updated**:
1. `test_tensor_dataset_negative_indexing()` - lines 159-162
2. `test_tensor_dataset_out_of_bounds()` - lines 180-183
3. `test_tensor_dataset_iteration_consistency()` - lines 209-212
4. `test_tensor_dataset_no_copy_on_access()` - lines 234-237
5. `test_tensor_dataset_memory_efficiency()` - lines 267-268

## Success Criteria

- [x] `TensorDataset` alias added to `datasets.mojo`
- [x] All `Tensor()` type usages replaced with `ExTensor`
- [x] All tests now compile without type errors
- [x] Proper use of `List[T]` initialization for tensor creation
- [x] Ownership transfer with `^` operator used correctly

## References

- [ExTensor Implementation](/shared/core/extensor.mojo)
- [Dataset Interface](/shared/data/datasets.mojo)
- [Mojo Syntax Standards](/CLAUDE.md#mojo-syntax-standards-v0257)

## Implementation Notes

- The `Tensor` type used in the test was undefined and likely a leftover from earlier development
- `ExTensor` is the correct type to use throughout the codebase
- The `TensorDataset` alias maintains backwards compatibility for any code that imports the old name
- All tensor initialization now properly uses `List[T]` with ownership transfer via `^`
