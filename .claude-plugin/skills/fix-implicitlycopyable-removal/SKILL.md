# Skill: Fix ImplicitlyCopyable Removal

## Overview

| Attribute | Value |
| --------- | ----- |
| **Date** | 2025-12-29 |
| **Session Context** | PR #2962 - ExTensor ImplicitlyCopyable Bug Fix |
| **Objective** | Remove ImplicitlyCopyable trait from struct with List fields and fix resulting compilation errors |
| **Outcome** | ✅ Success - 132 errors across 27 files resolved, all CI tests passing |
| **Duration** | Multiple iterations across several hours |
| **Files Modified** | 27 files (autograd, core, data, testing, training, layers) |

## When to Use This Skill

Use this skill when you need to:

1. **Remove ImplicitlyCopyable from a struct** that contains non-trivial fields
   (List, Dict, String, or other heap-allocated types)
2. **Fix compilation errors** after removing ImplicitlyCopyable trait from a widely-used type
3. **Debug memory corruption** caused by bitwise copies bypassing reference counting
4. **Systematically refactor** a codebase to use explicit copying instead of implicit copies

### Trigger Conditions

- Struct contains `List[T]`, `Dict[K,V]`, `String`, or other types with shared ownership
- Layer tests crash AFTER passing assertions (indicates destructor issues)
- Mojo compiler errors about "cannot implicitly copy" or "cannot transfer value"
- Memory corruption symptoms (use-after-free, double-free, segfaults)

## Problem Context

### The Bug

`ImplicitlyCopyable` on a struct with `List[Int]` fields causes Mojo to perform
**bitwise copies** that bypass `__copyinit__`. This breaks reference counting:

1. Implicit copy creates bitwise duplicate (refcount pointer copied, not incremented)
2. Refcount stays at 1 even with 2+ copies sharing the data
3. First destructor decrements refcount to 0 and frees memory
4. Second destructor accesses freed memory → **CRASH**

### Example: ExTensor Bug

```mojo
# BEFORE (BUGGY)
struct ExTensor(Copyable, ImplicitlyCopyable, Movable, Sized):
    var _shape: List[Int]      # Shared ownership
    var _strides: List[Int]    # Shared ownership
    var _refcount: UnsafePointer[Int]  # Manual refcount
```

Problem: Passing `ExTensor` by value triggers bitwise copy, not `__copyinit__`, so refcount isn't incremented.

## Verified Workflow

### Phase 1: Remove the Trait

1. **Identify the problematic struct** (usually contains List, Dict, or manual refcount)
2. **Remove ImplicitlyCopyable from trait list**:

   ```mojo
   # Change from:
   struct ExTensor(Copyable, ImplicitlyCopyable, Movable, Sized):

   # To:
   struct ExTensor(Copyable, Movable, Sized):
   ```

3. **Ensure `__copyinit__` exists and is correct**:

   ```mojo
   fn __copyinit__(out self, existing: Self):
       # MUST increment refcount for shared data
       self._data = existing._data
       self._shape = existing._shape.copy()  # Deep copy List
       self._strides = existing._strides.copy()
       self._refcount = existing._refcount
       if self._refcount:
           self._refcount[] += 1  # Critical!
   ```

### Phase 2: Add Explicit Copy Method

Create a `copy()` method using struct literal construction (REQUIRED because `__copyinit__` isn't directly callable):

```mojo
fn copy(self) -> Self:
    var result = Self {
        _data: self._data,
        _shape: self._shape.copy(),
        _strides: self._strides.copy(),
        _dtype: self._dtype,
        _numel: self._numel,
        _is_view: self._is_view,
        _refcount: self._refcount,
        _original_numel_quantized: self._original_numel_quantized,
        _allocated_size: self._allocated_size,
    }
    if result._refcount:
        result._refcount[] += 1  # Increment refcount
    return result^  # Transfer ownership
```

### Phase 3: Fix Compilation Errors Systematically

After removing ImplicitlyCopyable, the compiler will flag every implicit copy. Fix each pattern:

#### Pattern 1: List Indexing

```mojo
# BEFORE (implicit copy)
var tensor = tensor_list[i]

# AFTER (explicit copy)
var tensor = tensor_list[i].copy()
```

#### Pattern 2: Tuple Unpacking

```mojo
# BEFORE (doesn't work without ImplicitlyCopyable)
var images, labels = load_data()

# AFTER (use indexing)
var batch_data = load_data()
var images = batch_data[0].copy()
var labels = batch_data[1].copy()
```

#### Pattern 3: Function Returns

```mojo
# BEFORE (implicit copy on return)
fn get_tensor(self) -> ExTensor:
    return self._tensor  # Error!

# AFTER (explicit ownership transfer)
fn get_tensor(self) -> ExTensor:
    return self._tensor.copy()
```

#### Pattern 4: Ownership Transfer (Local Variables)

```mojo
# BEFORE
var result = some_tensor

# AFTER (transfer ownership, no copy needed)
var result = some_tensor^
```

#### Pattern 5: Tuple Returns

```mojo
# BEFORE
return (tensor1, tensor2)

# AFTER (transfer ownership)
return (tensor1^, tensor2^)
```

#### Pattern 6: Closure Captures

```mojo
# BEFORE (doesn't work - closures can't capture non-ImplicitlyCopyable)
var weights = get_weights()
fn forward(x: ExTensor) -> ExTensor:
    return conv2d(x, weights, ...)  # Error!

# AFTER (use UnsafePointer)
var weights_copy = get_weights().copy()
var weights_ptr = UnsafePointer.address_of(weights_copy)
fn forward(x: ExTensor) escaping -> ExTensor:
    return conv2d(x, weights_ptr[], ...)
```

#### Pattern 7: Function Parameters

```mojo
# BEFORE (var means owned/mutable)
fn argmax(var tensor: ExTensor) -> Int:
    ...

# AFTER (borrowed read-only)
fn argmax(tensor: ExTensor) -> Int:
    ...
```

### Phase 4: Batch Fix Strategy

For large codebases (27+ files), parallelize fixes by module:

1. **Autograd module** (backward_ops, functional, grad_utils, optimizers, tape_types, variable)
2. **Core module** (arithmetic, attention, conv, dtype_cast, extensor, lazy_eval, matrix, normalization, etc.)
3. **Data module** (_datasets_core, cache, transforms, loaders, etc.)
4. **Testing/Training modules** (layer_testers, models, metrics, gradient_clipping, mixed_precision)
5. **Utilities** (serialization, utils)

Use Task agents to fix files in parallel batches.

### Phase 5: Verification

```bash
# Local compilation check
pixi run mojo build shared/core/extensor.mojo

# Run affected tests
pixi run mojo test tests/shared/core/layers/test_dropout.mojo
pixi run mojo test tests/shared/core/layers/test_linear_struct.mojo

# Commit and push
git add -A
git commit -m "fix(shared): remove ImplicitlyCopyable from ExTensor, add explicit .copy()"
git push

# Monitor CI
gh pr checks <PR-number> --watch
```

## Failed Attempts (CRITICAL LEARNINGS)

### ❌ Attempt 1: Return self in copy() method

```mojo
fn copy(self) -> Self:
    return self  # FAILS - requires implicit copy
```

**Error**: `cannot implicitly copy 'ExTensor'`

**Why it failed**: Even though we're in a method that's supposed to copy, returning
`self` still requires an implicit copy, which we just removed.

### Attempt 2: Call `__copyinit__` directly (Failed)

```mojo
fn copy(self) -> Self:
    return Self.__copyinit__(self)  # FAILS - not callable
```

**Error**: `__copyinit__ is not directly callable`

**Why it failed**: Mojo doesn't allow direct calls to lifecycle methods like
`__copyinit__`, `__moveinit__`, etc. These are only invoked implicitly by the compiler.

### ❌ Attempt 3: Use tuple unpacking

```mojo
var images, labels = load_cifar10_batch(...)  # FAILS
```

**Error**: `cannot synthesize __copyinit__ for tuple containing non-ImplicitlyCopyable types`

**Why it failed**: Tuple destructuring requires each element to be ImplicitlyCopyable. Must use indexing instead.

**Solution**:

```mojo
var batch_data = load_cifar10_batch(...)
var images = batch_data[0].copy()
var labels = batch_data[1].copy()
```

### ❌ Attempt 4: Capture non-ImplicitlyCopyable in closure

```mojo
var weights = get_weights()
fn forward(x: ExTensor) -> ExTensor:
    return conv2d(x, weights, ...)  # FAILS
```

**Error**: `cannot synthesize __copyinit__ for closure capturing 'weights'`

**Why it failed**: Closures implicitly copy captured variables, which doesn't work without ImplicitlyCopyable.

**Solution**: Use UnsafePointer to capture by reference:

```mojo
var weights_copy = get_weights().copy()
var weights_ptr = UnsafePointer.address_of(weights_copy)
fn forward(x: ExTensor) escaping -> ExTensor:
    return conv2d(x, weights_ptr[], ...)
```

### ❌ Attempt 5: Assign without .copy()

```mojo
var pred_classes = predictions  # FAILS
```

**Error**: `cannot implicitly copy 'ExTensor'`

**Solution**:

```mojo
var pred_classes = predictions.copy()
```

## Results & Parameters

### Success Metrics

- **Files Modified**: 27 files across 5 modules
- **Compilation Errors Fixed**: 132 errors
- **CI Test Pass Rate**: 100% (45/45 test jobs passing)
- **Memory Corruption**: Eliminated (layer tests no longer crash)

### Files Changed

#### Core Changes

1. `shared/core/extensor.mojo` - Removed trait, added copy() method
2. `shared/core/layers/dropout.mojo` - Added ownership transfers
3. `shared/core/layers/linear.mojo` - Added ownership transfers

#### Data Pipeline

1. `shared/data/formats/cifar_loader.mojo` - Fixed tuple returns
2. `shared/data/datasets/cifar10.mojo` - Fixed tuple unpacking, added .copy()

#### Training/Testing

1. `shared/training/metrics/accuracy.mojo` - Fixed parameter signatures
2. `shared/training/metrics/base.mojo` - Added .copy() for returns
3. `shared/training/metrics/confusion_matrix.mojo` - Fixed parameter signatures
4. `shared/testing/layer_testers.mojo` - Fixed closure captures with UnsafePointer

#### Batch Fixes (Task Agent)

10-27. 18 additional files across autograd/, core/, data/, testing/, training/ modules

### Key Code Snippets

**ExTensor copy() method** (shared/core/extensor.mojo:449-474):

```mojo
fn copy(self) -> Self:
    var result = Self {
        _data: self._data,
        _shape: self._shape.copy(),
        _strides: self._strides.copy(),
        _dtype: self._dtype,
        _numel: self._numel,
        _is_view: self._is_view,
        _refcount: self._refcount,
        _original_numel_quantized: self._original_numel_quantized,
        _allocated_size: self._allocated_size,
    }
    if result._refcount:
        result._refcount[] += 1
    return result^
```

**Closure capture workaround** (shared/testing/layer_testers.mojo:605-612):

```mojo
var w_copy = weights.copy()
var b_copy = bias.copy()
var w_ptr = UnsafePointer.address_of(w_copy)
var b_ptr = UnsafePointer.address_of(b_copy)
fn forward(x: ExTensor) raises escaping -> ExTensor:
    return conv2d(x, w_ptr[], b_ptr[], stride=stride, padding=padding)
```

## Related Documentation

- [Mojo Lifetimes & Copy Semantics](https://docs.modular.com/mojo/manual/values/lifetimes/copy)
- [Mojo Ownership & Borrowing](https://docs.modular.com/mojo/manual/values/ownership)
- PR #2962: ExTensor ImplicitlyCopyable Removal

## Tags

`mojo` `ownership` `memory-safety` `refactoring` `ci-fix` `debugging` `reference-counting` `compilation-errors`
