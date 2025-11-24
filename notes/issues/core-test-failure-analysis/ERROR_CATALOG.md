# Core Test Suite - Complete Error Catalog

## Overview

This document provides the complete technical breakdown of all 57 compilation errors encountered when running the Core test suites. Organized by error type with specific line numbers and code examples.

---

## Error Type 1: ExTensor Initialization (8 errors)

### Error Message
```
__init__ method must return Self type with 'out' argument
```

### Severity: CRITICAL

### Locations
1. `/home/mvillmow/ml-odyssey/shared/core/extensor.mojo:89` - `__init__` method
2. `/home/mvillmow/ml-odyssey/shared/core/extensor.mojo:149` - `__copyinit__` method

Both methods appear in 4 test files each, creating 8 total errors (1 per inclusion).

### Current Code (extensor.mojo)
```mojo
# Line 89
fn __init__(mut self, shape: List[Int], dtype: DType) raises:
    """Initialize a new ExTensor with given shape and dtype."""
    # ... implementation

# Line 149
fn __copyinit__(mut self, existing: Self):
    """Copy constructor - creates shared ownership with reference counting."""
    # ... implementation
```

### Root Cause
Mojo v0.25.7+ deprecated `mut self` in initialization methods. The required syntax is `out self` which explicitly indicates ownership is being transferred to the new object being initialized.

### Fix
Replace `mut self` with `out self`:
```mojo
# Line 89 - FIXED
fn __init__(out self, shape: List[Int], dtype: DType) raises:
    """Initialize a new ExTensor with given shape and dtype."""
    # ... rest unchanged

# Line 149 - FIXED
fn __copyinit__(out self, existing: Self):
    """Copy constructor - creates shared ownership with reference counting."""
    # ... rest unchanged
```

### Why This Matters
Without `out`, Mojo cannot properly track ownership. The `out` parameter means:
- The method takes ownership of `self`
- The method must initialize all fields of `self`
- After return, `self` is properly constructed

---

## Error Type 2: DType Not Comparable (17 errors)

### Error Message
```
invalid call to 'assert_equal': failed to infer parameter 'T', argument type 'DType'
does not conform to trait 'Comparable'
```

### Severity: HIGH

### Count: 17 instances

### Specific Locations in test_tensors.mojo
```
Line 172:  assert_equal(y.dtype(), DType.float32)
Line 194:  assert_equal(y.dtype(), DType.float64)
Line 214:  assert_equal(y.dtype(), DType.float32)
Line 247:  assert_equal(t_f32.dtype(), DType.float32)
Line 250:  assert_equal(t_f64.dtype(), DType.float64)
Line 253:  assert_equal(t_i32.dtype(), DType.int32)
Line 311:  assert_equal(t_f16.dtype(), DType.float16)
Line 316:  assert_equal(t_f32.dtype(), DType.float32)
Line 321:  assert_equal(t_f64.dtype(), DType.float64)
Line 332:  assert_equal(t_i8.dtype(), DType.int8)
Line 337:  assert_equal(t_i16.dtype(), DType.int16)
Line 342:  assert_equal(t_i32.dtype(), DType.int32)
Line 347:  assert_equal(t_i64.dtype(), DType.int64)
Line 358:  assert_equal(t_u8.dtype(), DType.uint8)
Line 363:  assert_equal(t_u16.dtype(), DType.uint16)
Line 368:  assert_equal(t_u32.dtype(), DType.uint32)
Line 373:  assert_equal(t_u64.dtype(), DType.uint64)
```

### Root Cause
The generic assertion function `assert_equal[T: Comparable]` requires type T to conform to the `Comparable` trait. `DType` is an enum that does not conform to this trait (Mojo enums don't automatically implement comparison traits).

### Original assert_equal Definition
Location: `/home/mvillmow/ml-odyssey/tests/shared/conftest.mojo:46`
```mojo
fn assert_equal[T: Comparable](a: T, b: T, message: String = "") raises:
    """Assert exact equality of two values."""
    if a != b:
        var error_msg = message if message else "Values are not equal"
        raise Error(error_msg)
```

### Solution 1: Add Specialized DType Function

Add to `conftest.mojo` after line 78:
```mojo
fn assert_dtype_equal(a: DType, b: DType, message: String = "") raises:
    """Assert exact equality of DType values."""
    if a != b:
        var error_msg = message if message else "DTypes are not equal"
        raise Error(error_msg)
```

Then update all 17 calls in `test_tensors.mojo`:
```mojo
# BEFORE
assert_equal(y.dtype(), DType.float32)

# AFTER
assert_dtype_equal(y.dtype(), DType.float32)
```

### Solution 2: Make DType Comparable (Not Recommended)

This would require modifying core Mojo types which is not practical. Solution 1 is preferred.

---

## Error Type 3: ExTensor Not ImplicitlyCopyable (7 errors)

### Error Message
```
value of type 'ExTensor' cannot be implicitly copied, it does not conform to
'ImplicitlyCopyable'
```

### Severity: HIGH

### Count: 7 instances

### Specific Locations

**test_activations.mojo:**
```
Line 206:  fn forward(inp: ExTensor) raises -> ExTensor:
Line 213:  fn backward_input(grad: ExTensor, inp: ExTensor) raises -> ExTensor:
```

**gradient_checking.mojo (included from test_activations.mojo):**
```
Line 221:  fn scaled_forward(inp: ExTensor) raises -> ExTensor:
```

**activation.mojo (included indirectly):**
```
Line 1088: return result
Line 1259: return result
```

**Related cascading errors from type synthesis:**
```
test_activations.mojo:206  - cannot synthesize __copyinit__ (field 'field0')
test_activations.mojo:213  - cannot synthesize __copyinit__ (field 'field0')
gradient_checking.mojo:221 - cannot synthesize __copyinit__ (field 'field1')
```

### Root Cause

ExTensor is declared as:
```mojo
struct ExTensor(Copyable, Movable):
```

This means ExTensor can be copied, but Mojo v0.25.7+ distinguishes between:
- **`Copyable`**: Can be explicitly copied via `.copy()` method or `^` operator
- **`ImplicitlyCopyable`**: Can be copied implicitly (like built-in types)

When a function parameter is `inp: ExTensor` without modifiers, Mojo assumes implicit copying. Without `ImplicitlyCopyable`, this is not allowed.

### Detailed Error for Line 206

```
error: value of type 'ExTensor' cannot be implicitly copied,
it does not conform to 'ImplicitlyCopyable'
    fn forward(inp: ExTensor) raises -> ExTensor:
       ^~~~~~~

note: consider transferring the value with '^'
    fn forward(inp: ExTensor) raises -> ExTensor:
             ^

note: you can copy it explicitly with '.copy()'
    fn forward(inp: ExTensor) raises -> ExTensor:
             .copy()
```

### Solutions

**Solution 1: Add ImplicitlyCopyable Trait (RECOMMENDED)**

File: `shared/core/extensor.mojo:43`

**BEFORE:**
```mojo
struct ExTensor(Copyable, Movable):
```

**AFTER:**
```mojo
struct ExTensor(Copyable, Movable, ImplicitlyCopyable):
```

This is the simplest fix. ExTensor already implements copy semantics through `__copyinit__`, so it's safe to mark as implicitly copyable.

**Solution 2: Use Ownership Transfer (NOT RECOMMENDED)**

Modify function signatures to use `^` operator:
```mojo
fn forward(inp: ExTensor^) raises -> ExTensor:
    # inp is moved, caller loses access
```

This would require changes throughout the codebase and complicates API.

**Solution 3: Use Reference Semantics (NOT RECOMMENDED)**

Use borrowed references:
```mojo
fn forward(ref inp: ExTensor) raises -> ExTensor:
```

This changes the API contract.

### Recommended Fix
Update line 43 of `extensor.mojo` to add `ImplicitlyCopyable` trait. This enables seamless parameter passing and return values.

---

## Error Type 4: exp() Type Inference (4 errors)

### Error Message
```
no matching function in call to 'exp'
```

### Severity: MEDIUM

### Count: 4 instances

### Specific Locations

**activation.mojo:**
```
Line 1008: var exp_neg_abs = exp(neg_x_abs)  # In selu_forward()
           ~~~^~~~~~~~~~~

Line 1168: var exp_neg_abs = exp(neg_x_abs)  # In celu_forward()
           ~~~^~~~~~~~~~~
```

### Full Error Details

```
error: no matching function in call to 'exp'
    var exp_neg_abs = exp(neg_x_abs)
                      ~~~^~~~~~~~~~~

note: candidate not viable: failed to infer parameter 'dtype',
it isn't used in any argument
    """Tests for activation functions."""
    ^

note: candidate not viable: failed to infer parameter 'T',
argument type 'ExTensor' does not conform to trait '_Expable'
    """Tests for activation functions."""
    ^
```

### Root Cause

The `exp()` function in `elementwise.mojo:86` expects an `ExTensor`:
```mojo
fn exp(tensor: ExTensor) raises -> ExTensor:
    """Exponential function element-wise (e^x)."""
    return dispatch_float_unary[_exp_op](tensor)
```

The variables `neg_x_abs` (lines 1008, 1168) are ExTensor results from operations like:
```mojo
var neg_x_abs = -(abs(x))  # Line 1007 in selu_forward
var neg_x_abs = -(abs(x))  # Line 1167 in celu_forward
```

However, the type inference for the `-()` operator on an ExTensor result is not properly resolved.

### Context - selu_forward (activation.mojo:1001-1090)

```mojo
fn selu_forward(x: ExTensor, alpha: Float64 = 1.67326324, scale: Float64 = 1.0507009) raises -> ExTensor:
    """SELU forward pass."""
    var neg_x_abs = -(abs(x))     # Line 1007 - Type: ExTensor (implicit)
    var exp_neg_abs = exp(neg_x_abs)  # Line 1008 - ERROR: Cannot infer type
    # ... rest of function
```

### Context - celu_forward (activation.mojo:1161-1260)

```mojo
fn celu_forward(x: ExTensor, alpha: Float64 = 1.0) raises -> ExTensor:
    """CELU forward pass."""
    var neg_x_abs = -(abs(x))     # Line 1167 - Type: ExTensor (implicit)
    var exp_neg_abs = exp(neg_x_abs)  # Line 1168 - ERROR: Cannot infer type
    # ... rest of function
```

### Solution

Add explicit type annotation:
```mojo
# BEFORE
var neg_x_abs = -(abs(x))
var exp_neg_abs = exp(neg_x_abs)

# AFTER - with explicit type
var abs_neg_x: ExTensor = abs(x)
var neg_x_abs: ExTensor = -(abs_neg_x)
var exp_neg_abs: ExTensor = exp(neg_x_abs)
```

Or simplify to one line:
```mojo
var exp_neg_abs = exp(-(abs(x)))
```

### Why This Happens

Mojo's type inference for operator results on custom types (like ExTensor) sometimes requires explicit declaration. The `-()` operator on ExTensor returns an ExTensor, but without explicit type, the compiler struggles with parameter inference for `exp()`.

---

## Error Type 5: Float64 vs Float32 Assertion Mismatch (4 errors)

### Error Message
```
invalid call to 'assert_almost_equal': argument #0 cannot be converted from
'Float64' to 'Float32'
```

### Severity: MEDIUM

### Count: 4 instances

### Specific Locations

**test_activations.mojo:**
```
Line 722: assert_almost_equal(y._data.bitcast[Float64]()[0], 0.0, tolerance=1e-10)
Line 723: assert_almost_equal(y._data.bitcast[Float64]()[1], 1.0, tolerance=1e-10)
Line 736: assert_almost_equal(y._data.bitcast[Float64]()[0], 0.5, tolerance=1e-10)
```

**test_tensors.mojo:**
```
Line 198: assert_almost_equal(y._data.bitcast[Float64]()[i], 1.0, tolerance=1e-10)
```

### Current assert_almost_equal Definition

Location: `tests/shared/conftest.mojo:80`
```mojo
fn assert_almost_equal(
    a: Float32, b: Float32, tolerance: Float32 = 1e-6, message: String = ""
) raises:
    """Assert floating point near-equality."""
    var diff = abs(a - b)
    if diff > tolerance:
        var error_msg = message if message else (
            String(a) + " !≈ " + String(b) + " (diff: " + String(diff) + ")"
        )
        raise Error(error_msg)
```

### Root Cause

The function is hardcoded to work with `Float32` parameters. The test code is comparing `Float64` values (from `bitcast[Float64]()`).

Mojo's type system is strict: `Float64` cannot be implicitly converted to `Float32`, so the function call fails.

### Solution: Add Float64 Overload

Add to `conftest.mojo` after the existing `assert_almost_equal`:
```mojo
fn assert_almost_equal(
    a: Float64, b: Float64, tolerance: Float64 = 1e-6, message: String = ""
) raises:
    """Assert floating point near-equality for Float64."""
    var diff = abs(a - b)
    if diff > tolerance:
        var error_msg = message if message else (
            String(a) + " !≈ " + String(b) + " (diff: " + String(diff) + ")"
        )
        raise Error(error_msg)
```

The test code can then remain unchanged - Mojo's overload resolution will select the correct version.

### Why Both Versions

Different contexts require different precision:
- `Float32`: Standard assertions for float32 tensors (precision: ~1e-6)
- `Float64`: Higher precision assertions for double-precision comparisons (precision: ~1e-10)

---

## Error Type 6: Function Signature Mismatches (2 errors)

### Error 6a: elu_backward Parameter Mismatch

### Error Message
```
invalid call to 'elu_backward': argument passed both as positional and keyword
operand: 'alpha'
```

**Location**: `test_activations.mojo:700`

**Code**:
```mojo
return elu_backward(grad, inp, out, alpha=1.0)
                                      ^^^^^^^^^ Error here
```

**Function Definition** (activation.mojo:1192):
```mojo
fn elu_backward(grad_output: ExTensor, x: ExTensor, alpha: Float64 = 1.0) raises -> ExTensor:
```

**Root Cause**: The function signature has 3 parameters:
1. `grad_output`
2. `x`
3. `alpha` (with default)

But the test passes 4 arguments:
1. `grad` → `grad_output`
2. `inp` → `x`
3. `out` → ??? (not in signature)
4. `alpha=1.0` → `alpha`

The `out` argument doesn't exist in the function signature.

**Fix**:
```mojo
# BEFORE (WRONG)
return elu_backward(grad, inp, out, alpha=1.0)

# AFTER (CORRECT)
return elu_backward(grad, inp, alpha=1.0)
```

### Error 6b: check_gradient Closure Signature

### Error Message
```
invalid call to 'check_gradient': argument #0 cannot be converted from
'fn(inp: ExTensor) raises escaping -> ExTensor' to
'fn(ExTensor) raises -> ExTensor'
```

**Location**: `test_activations.mojo:217`

**Code**:
```mojo
fn forward(inp: ExTensor) raises -> ExTensor:
    return tanh(inp)

fn backward_input(grad: ExTensor, inp: ExTensor) raises -> ExTensor:
    var result = tanh_backward(grad, inp)
    return result[0]

check_gradient(forward, backward_input, x, grad_out, rtol=1e-4, atol=1e-7)
               ^^^^^^^ Error: Type mismatch with nested function
```

**check_gradient Definition** (gradient_checking.mojo:178):
```mojo
fn check_gradient(
    forward: fn(ExTensor) raises -> ExTensor,
    backward_input: fn(ExTensor, ExTensor) raises -> ExTensor,
    # ...
) raises:
```

**Root Cause**: The nested functions are defined inside the test function, capturing local variables. This makes them "escaping" closures with different type signatures than standalone functions.

Mojo's function type system distinguishes:
- `fn(ExTensor) raises -> ExTensor` - standalone function
- `fn(inp: ExTensor) raises escaping -> ExTensor` - nested function with captures

The `escaping` keyword indicates the function escapes the local scope.

**Fix**: Either:

1. Define functions at module level (not nested)
2. Change `check_gradient` to accept escaping functions
3. Use a struct-based approach instead of nested functions

---

## Error Type 7: Scalar abs() Wrong Function (4 errors)

### Error Message
```
invalid call to 'abs': argument #0 cannot be converted from 'Float32/Float64'
to 'ExTensor'
```

### Severity: MEDIUM

### Count: 4 instances

### Specific Locations

**elementwise.mojo:30-32** - In `_abs_op` function:
```mojo
@always_inline
fn _abs_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Absolute value operation."""
    @parameter
    if T == DType.float16 or T == DType.float32:
        return Scalar[T](abs(Float32(x)))  # Line 30 - ERROR
    else:
        return Scalar[T](abs(Float64(x)))  # Line 32 - ERROR
```

### Root Cause

The code is calling `abs(Float32(x))` where:
- `Float32(x)` creates a Float32 scalar
- But `abs()` is overloaded to expect `ExTensor` (from elementwise.mojo:35)

```mojo
# This abs() expects ExTensor:
fn abs(tensor: ExTensor) raises -> ExTensor:
    return dispatch_unary[_abs_op](tensor)
```

So calling `abs(Float32(...))` fails because Float32 is not ExTensor.

### Solution

Use Mojo's built-in `abs()` for scalars or implement scalar absolute value directly:

**Option 1: Direct Implementation (RECOMMENDED)**
```mojo
@always_inline
fn _abs_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Absolute value operation."""
    if x >= Scalar[T](0):
        return x
    else:
        return -x
```

**Option 2: Use Math Library**
```mojo
from math import abs as math_abs

@always_inline
fn _abs_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Absolute value operation."""
    @parameter
    if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_abs(Float32(x)))
    else:
        return Scalar[T](math_abs(Float64(x)))
```

**Option 3: Use Scalar Method**
```mojo
# Mojo scalars have built-in abs
return x.abs()
```

### Why This Happens

Overloading `abs()` for both `ExTensor` and scalars without proper disambiguation causes conflicts. The direct implementation approach is cleanest.

---

## Error Type 8: Type System Synthesis (3 errors)

### Error 8a: Cannot Synthesize __copyinit__

### Error Message
```
cannot synthesize __copyinit__ because field 'field0' has non-copyable type
'ExTensor'
```

### Severity: LOW (symptom of Error Type 3)

### Locations
```
test_activations.mojo:206  - forward function
test_activations.mojo:213  - backward_input function
gradient_checking.mojo:221 - scaled_forward function
```

### Root Cause
When nested functions are defined with ExTensor parameters, Mojo tries to create a wrapper struct to capture state. Since ExTensor is not `ImplicitlyCopyable`, the struct cannot auto-generate `__copyinit__`.

This is a cascading error from Error Type 3.

### Fix
Adding `ImplicitlyCopyable` to ExTensor (Error Type 3 fix) resolves this automatically.

### Error 8b: GradientPair Not Subscriptable

### Error Message
```
'GradientPair' is not subscriptable, it does not implement the
`__getitem__`/`__setitem__` methods
```

### Severity: LOW

### Location: `test_activations.mojo:215`

**Code**:
```mojo
fn backward_input(grad: ExTensor, inp: ExTensor) raises -> ExTensor:
    var result = tanh_backward(grad, inp)
    return result[0]  # ERROR: Cannot subscript GradientPair
               ^^^
```

### Root Cause
`tanh_backward()` returns a `GradientPair` (presumably a struct or tuple with multiple gradient values). The test code assumes it's subscriptable like a tuple, but `GradientPair` doesn't implement `__getitem__`.

### Solution
Access the gradient by field name instead of index:
```mojo
# BEFORE
return result[0]

# AFTER - need to check GradientPair definition
return result.gradient  # or whatever the field is named
```

Need to find `GradientPair` definition to determine correct field name.

---

## Summary Table

| Error Type | Count | Severity | Files | Quick Fix |
|------------|-------|----------|-------|-----------|
| 1. __init__ syntax | 8 | CRITICAL | extensor.mojo | Change `mut` → `out` |
| 2. DType not Comparable | 17 | HIGH | test_tensors.mojo | Add assert_dtype_equal |
| 3. Not ImplicitlyCopyable | 7 | HIGH | extensor.mojo | Add trait |
| 4. exp() type inference | 4 | MEDIUM | activation.mojo | Add type hints |
| 5. Float64 vs Float32 | 4 | MEDIUM | conftest.mojo | Add overload |
| 6. Function signatures | 2 | MEDIUM | test_activations.mojo | Fix args |
| 7. Scalar abs() | 4 | MEDIUM | elementwise.mojo | Direct impl |
| 8. Type synthesis | 3 | LOW | auto-fixed | Fix #3 |
| **TOTAL** | **57** | - | - | - |

---

## Test Execution Order After Fixes

1. Compile `extensor.mojo` (fixes types 1, 3, 8a)
2. Compile `conftest.mojo` (fixes type 5, 2)
3. Compile `elementwise.mojo` (fixes type 7)
4. Compile `activation.mojo` (fixes type 4)
5. Run `test_layers.mojo`
6. Run `test_tensors.mojo` (requires type 2 fix)
7. Run `test_activations.mojo` (requires types 4, 5, 6, 8b fixes)
8. Run `test_advanced_activations.mojo`

