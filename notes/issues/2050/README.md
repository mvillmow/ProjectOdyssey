# Issue #2050: Type Mismatch in TensorDataset

## Objective

Fix Float32/Float64 type mismatch in test_tensor_dataset.mojo where ExTensor stores Float32 but returns Float64.

## Problem Analysis

### Current Behavior

- ExTensor stores data as **Float32** internally (line 230 in extensor.mojo)
- ExTensor.__getitem__() returns **Float64** (line 570 in extensor.mojo)
- Tests expect **Float64** when accessing tensor elements (lines 166, 170, 244, etc.)

### Root Cause

The `__getitem__` method unconditionally returns `Float64` regardless of the tensor's actual dtype. For ML workloads, Float32 is the standard and preferred precision:

- **Memory efficiency**: Float32 uses 4 bytes vs Float64's 8 bytes (2x savings)
- **Performance**: Most neural networks use Float32 for training
- **Hardware optimization**: GPUs and accelerators optimized for Float32
- **Consistency**: Should match stored dtype when possible

### Solution

Change ExTensor to consistently use Float32 for floating-point operations:

1. Update `ExTensor.__getitem__()` to return Float32 instead of Float64
2. Update test expectations from Float64 to Float32
3. Ensure all conversion/access methods use consistent dtype

## Files Changed

- `/home/mvillmow/ml-odyssey/shared/core/extensor.mojo` - Change `__getitem__` return type
- `/home/mvillmow/ml-odyssey/tests/shared/data/datasets/test_tensor_dataset.mojo` - Update assertions

## Success Criteria

- [x] ExTensor.__getitem__() returns Float32
- [x] All test assertions updated to expect Float32
- [x] Tests pass without type mismatches
- [x] No other ExTensor functionality broken

## Changes Made

### 1. ExTensor (/shared/core/extensor.mojo)

- **Line 570**: Changed `__getitem__` return type from `Float64` to `Float32`
- **Line 577**: Updated docstring to reflect Float32 return type
- **Line 590**: Changed implementation to call `_get_float32()` instead of `_get_float64()`

### 2. Test File (/tests/shared/data/datasets/test_tensor_dataset.mojo)

Updated all assertions from `Float64` to `Float32`:
- Lines 166, 170: `test_tensor_dataset_negative_indexing()`
- Lines 219-222: `test_tensor_dataset_iteration_consistency()`
- Lines 246, 251: `test_tensor_dataset_no_copy_on_access()`
- Lines 278, 282, 286: `test_tensor_dataset_memory_efficiency()`

## Rationale

Float32 is the correct choice for ML workloads because:
1. **Standard for neural networks**: Industry-standard precision for training and inference
2. **Memory efficient**: 2x smaller than Float64 (4 bytes vs 8 bytes per element)
3. **Hardware optimized**: GPUs and accelerators have native Float32 support
4. **Performance**: Faster computation with single-precision arithmetic
5. **Consistency**: Matches stored dtype throughout the tensor lifecycle

## Implementation Notes

The fix prioritizes Float32 because:
1. **ML standard**: Most neural networks use Float32
2. **Memory efficient**: 2x savings vs Float64
3. **Hardware aligned**: GPUs optimized for Float32
4. **Performance**: Faster computation with single-precision
