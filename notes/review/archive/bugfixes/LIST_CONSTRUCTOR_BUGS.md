# List Constructor Bugs - Comprehensive Audit

## Problem

The pattern `var list = List[Int](n)` followed by `list[i] = value` is **unsafe** and causes memory corruption.

## Affected Files

### 1. shared/core/matrix.mojo (FIXED ✅)

- Line 213: `var input_strides = List[Int](ndim)` - **FIXED**
- Line 229: `var result_coords = List[Int](ndim)` - **FIXED**
- Line 236: `var input_coords = List[Int](ndim)` - **FIXED**

### 2. shared/core/shape.mojo (NEEDS FIX ❌)

- Line 48: `var final_shape = List[Int](new_len)` + line 59: `final_shape[i] = inferred_size`
- Line 141: `var new_shape = List[Int](new_dims)` + line 145: `new_shape[j] = old_shape[i]`
- Line 177: `var new_shape = List[Int](new_ndim)` + line 181/183: `new_shape[i] = ...`
- Line 296: `var result_shape = List[Int](ndim)` + line 299/301: `result_shape[i] = ...`

### 3. shared/training/metrics/accuracy.mojo (NEEDS FIX ❌)

- Line 118: `var result_shape = List[Int](batch_size)` - check usage
- Line 348: `var result_shape = List[Int](num_classes)` - check usage

### 4. shared/training/metrics/confusion_matrix.mojo (NEEDS FIX ❌)

- Line 323: `var result_shape = List[Int](batch_size)` - check usage

### 5. shared/training/trainer_interface.mojo (NEEDS FIX ❌)

- Line 270: `var batch_labels_shape = List[Int](actual_batch_size)` - check usage

## Safe Patterns

### Pattern 1: Build with append (RECOMMENDED)

```mojo
var list = List[Int]()
for i in range(n):
    list.append(value)
```text

### Pattern 2: Initialize then assign

```mojo
var list = List[Int]()
for _ in range(n):
    list.append(0)  # Initialize with placeholder
for i in range(n):
    list[i] = actual_value  # Now safe to index
```text

### Pattern 3: Build without indexing

```mojo
var list = List[Int]()
for item in items:
    list.append(compute_value(item))  # Append directly
```text

## Unsafe Patterns (DO NOT USE)

### Anti-pattern 1: Assume List[Int](n) creates n elements

```mojo
var list = List[Int](n)  # BAD: May not have n elements!
list[0] = value  # BAD: May crash!
```text

### Anti-pattern 2: Index without append

```mojo
var list = List[Int]()
list[0] = value  # BAD: List is empty!
```text

## Priority for Fixing

1. **CRITICAL (causes crashes in production)**:
   - shared/core/shape.mojo - Used in reshape operations
   - These will crash when called!

2. **HIGH (may cause crashes)**:
   - shared/training/metrics/* - Used during evaluation
   - shared/training/trainer_interface.mojo - Used in training loop

3. **ALREADY FIXED**:
   - shared/core/matrix.mojo - transpose() function

## Action Items

1. ✅ Fix shared/core/matrix.mojo (DONE)
2. ❌ Fix shared/core/shape.mojo (4 instances)
3. ❌ Audit and fix shared/training/metrics/accuracy.mojo (2 instances)
4. ❌ Audit and fix shared/training/metrics/confusion_matrix.mojo (1 instance)
5. ❌ Audit and fix shared/training/trainer_interface.mojo (1 instance)
6. ❌ Add linting rule to detect this pattern
7. ❌ Add unit tests for all fixed functions

## Testing Strategy

For each fix:

1. Create a failing test that reproduces the crash
2. Apply the fix
3. Verify the test passes
4. Run integration tests

## Notes

- The List[Int](n) constructor behavior is implementation-dependent
- Never assume list has elements after construction
- Always use append() or verify length before indexing
- This is a common source of memory corruption bugs
