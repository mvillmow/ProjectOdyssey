# Quick Fix Guide - Training Test Suite Errors

## Command to Run Tests

```bash
pixi run mojo -I . tests/shared/training/test_optimizers.mojo
pixi run mojo -I . tests/shared/training/test_training_loop.mojo
```

## Top Priority Fixes (Do These First)

### Fix 1: simdwidthof Import (5 minutes)
**File:** `/home/mvillmow/ml-odyssey/shared/core/arithmetic_simd.mojo:27`

```diff
- from sys.info import simdwidthof
+ from sys import simdwidthof
```

### Fix 2: Add ImplicitlyCopyable Trait (5 minutes)
**File:** `/home/mvillmow/ml-odyssey/shared/core/extensor.mojo:43`

```diff
- struct ExTensor(Copyable, Movable):
+ struct ExTensor(Copyable, Movable, ImplicitlyCopyable):
```

### Fix 3: __init__ Signature (5 minutes)
**File:** `/home/mvillmow/ml-odyssey/shared/core/extensor.mojo:89`

```diff
- fn __init__(mut self, shape: List[Int], dtype: DType) raises:
+ fn __init__(out self, shape: List[Int], dtype: DType) raises:
```

### Fix 4: __copyinit__ Signature (5 minutes)
**File:** `/home/mvillmow/ml-odyssey/shared/core/extensor.mojo:149`

```diff
- fn __copyinit__(mut self, existing: Self):
+ fn __copyinit__(out self, existing: Self):
```

### Fix 5: Base.mojo __init__ (5 minutes)
**File:** `/home/mvillmow/ml-odyssey/shared/training/base.mojo:69`

```diff
- fn __init__(
+ fn __init__(out
      mut self,
```

### Fix 6: RMSprop - Remove 'owned' (10 minutes)
**File:** `/home/mvillmow/ml-odyssey/shared/training/optimizers/rmsprop.mojo:40`

```diff
- owned buf: ExTensor = zeros(List[Int](0), DType.float32)
+ var buf: ExTensor
```

Then update function signature to require explicit buf parameter (remove default).

## Affected Files Summary

### Will Fail Until Fixed:
- shared/core/extensor.mojo (fixes 1-4 needed)
- shared/core/arithmetic_simd.mojo (fix 1 needed)
- shared/training/base.mojo (fix 5 needed)
- shared/training/optimizers/rmsprop.mojo (fix 6 needed)
- shared/training/optimizers/sgd.mojo (will work after fix 2)
- shared/training/optimizers/adam.mojo (will work after fix 2)

### Test Files:
- test_optimizers.mojo (needs fixes 1-2 in core, then will pass)
- test_training_loop.mojo (needs implementation of TrainingLoop, MSELoss, etc.)

## Verification Steps

After applying CRITICAL fixes:

```bash
# Should compile successfully
pixi run mojo -I . shared/core/extensor.mojo

# Should compile successfully  
pixi run mojo -I . shared/training/optimizers/sgd.mojo

# Will still fail (missing TrainingLoop class)
pixi run mojo -I . tests/shared/training/test_optimizers.mojo
```

## Error Messages You'll See

### Before Fixes:
```
error: module 'info' does not contain 'simdwidthof'
error: value of type 'ExTensor' cannot be implicitly copied
error: __init__ method must return Self type with 'out' argument
error: 'owned' has been deprecated, use 'var' instead
```

### After Critical Fixes:
```
error: use of unknown declaration 'TrainingLoop'
error: use of unknown declaration 'create_simple_model'
error: use of unknown declaration 'assert_greater'
```

These indicate missing implementations (separate issue).

## Minimal Reproducible Example

Test the core fix with:

```mojo
from shared.core.extensor import ExTensor, zeros

fn test_extensor_copy() raises:
    var a = zeros(List[Int](2, 3), DType.float32)
    var b = a  # This should work after Fix 2
    print("Copy successful!")

test_extensor_copy()
```

If this compiles after your fixes, you're on the right track!
