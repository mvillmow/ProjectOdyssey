# Core Test Suite Fix - Quick Start Guide

## TL;DR

All Core test suites failed at compilation due to 5 critical/high priority issues. **26 minutes of fixes needed** to get tests running.

| Priority | Issue | Files | Time |
|----------|-------|-------|------|
| 🔴 CRITICAL | ExTensor.__init__ using `mut` instead of `out` | extensor.mojo:89,149 | 2 min |
| 🔴 CRITICAL | ExTensor not ImplicitlyCopyable | extensor.mojo:43 | 1 min |
| 🟠 HIGH | DType not Comparable | conftest.mojo + test_tensors.mojo | 10 min |
| 🟠 HIGH | Float64 assert overload missing | conftest.mojo | 5 min |
| 🟠 HIGH | Scalar abs() wrong function | elementwise.mojo:25-32 | 3 min |
| 🟡 MEDIUM | exp() type inference | activation.mojo:1008,1168 | 5 min |
| 🟡 MEDIUM | Function signature mismatches | test_activations.mojo | 2 min |

---

## Fix Checklist (In Order)

### Step 1: Fix ExTensor (3 min) - CRITICAL

**File**: `/home/mvillmow/ml-odyssey/shared/core/extensor.mojo`

#### Change 1: Line 89
```mojo
# BEFORE
fn __init__(mut self, shape: List[Int], dtype: DType) raises:

# AFTER
fn __init__(out self, shape: List[Int], dtype: DType) raises:
```

#### Change 2: Line 149
```mojo
# BEFORE
fn __copyinit__(mut self, existing: Self):

# AFTER
fn __copyinit__(out self, existing: Self):
```

#### Change 3: Line 43
```mojo
# BEFORE
struct ExTensor(Copyable, Movable):

# AFTER
struct ExTensor(Copyable, Movable, ImplicitlyCopyable):
```

✅ Verify: `pixi run mojo -I . shared/core/extensor.mojo` should compile

---

### Step 2: Add Test Helper Functions (5 min) - HIGH

**File**: `/home/mvillmow/ml-odyssey/tests/shared/conftest.mojo`

#### Add after existing assert_almost_equal (around line 99):

```mojo
fn assert_dtype_equal(a: DType, b: DType, message: String = "") raises:
    """Assert exact equality of DType values."""
    if a != b:
        var error_msg = message if message else "DTypes are not equal"
        raise Error(error_msg)


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

✅ Verify: `pixi run mojo -I . tests/shared/conftest.mojo` should compile

---

### Step 3: Fix Scalar abs() (3 min) - HIGH

**File**: `/home/mvillmow/ml-odyssey/shared/core/elementwise.mojo`

#### Replace lines 25-32:

```mojo
# BEFORE
@always_inline
fn _abs_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Absolute value operation."""
    @parameter
    if T == DType.float16 or T == DType.float32:
        return Scalar[T](abs(Float32(x)))
    else:
        return Scalar[T](abs(Float64(x)))

# AFTER
@always_inline
fn _abs_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Absolute value operation."""
    if x >= Scalar[T](0):
        return x
    else:
        return -x
```

✅ Verify: `pixi run mojo -I . shared/core/elementwise.mojo` should compile

---

### Step 4: Fix exp() Type Inference (5 min) - MEDIUM

**File**: `/home/mvillmow/ml-odyssey/shared/core/activation.mojo`

#### Fix 1: Around line 1007-1008

```mojo
# BEFORE
var neg_x_abs = -(abs(x))
var exp_neg_abs = exp(neg_x_abs)

# AFTER
var neg_x_abs: ExTensor = -(abs(x))
var exp_neg_abs: ExTensor = exp(neg_x_abs)
```

#### Fix 2: Around line 1167-1168

```mojo
# BEFORE
var neg_x_abs = -(abs(x))
var exp_neg_abs = exp(neg_x_abs)

# AFTER
var neg_x_abs: ExTensor = -(abs(x))
var exp_neg_abs: ExTensor = exp(neg_x_abs)
```

✅ Verify: `pixi run mojo -I . shared/core/activation.mojo` should compile

---

### Step 5: Update test_tensors.mojo (10 min) - MEDIUM

**File**: `/home/mvillmow/ml-odyssey/tests/shared/core/test_tensors.mojo`

Replace all calls to `assert_equal(y.dtype(), DType.xxx)` with `assert_dtype_equal(y.dtype(), DType.xxx)`

**Lines to update**: 172, 194, 214, 247, 250, 253, 311, 316, 321, 332, 337, 342, 347, 358, 363, 368, 373

Example:
```mojo
# BEFORE (Line 172)
assert_equal(y.dtype(), DType.float32)

# AFTER
assert_dtype_equal(y.dtype(), DType.float32)
```

Search & Replace:
```bash
# In your editor, find: assert_equal(.*\.dtype\(\)
# Replace with: assert_dtype_equal($1
```

✅ Verify: `pixi run mojo -I . tests/shared/core/test_tensors.mojo` should compile

---

### Step 6: Fix test_activations.mojo (2 min) - MEDIUM

**File**: `/home/mvillmow/ml-odyssey/tests/shared/core/test_activations.mojo`

#### Fix 1: Line 700

```mojo
# BEFORE
return elu_backward(grad, inp, out, alpha=1.0)

# AFTER
return elu_backward(grad, inp, alpha=1.0)
```

#### Fix 2: Line 215 (Optional - if GradientPair.gradient is the field name)

```mojo
# BEFORE
return result[0]

# AFTER
return result.gradient  # Check GradientPair definition for correct field name
```

✅ Verify: `pixi run mojo -I . tests/shared/core/test_activations.mojo` should compile

---

### Step 7: Verify All Tests Pass

Run each test:
```bash
pixi run mojo -I . tests/shared/core/test_layers.mojo
pixi run mojo -I . tests/shared/core/test_activations.mojo
pixi run mojo -I . tests/shared/core/test_advanced_activations.mojo
pixi run mojo -I . tests/shared/core/test_tensors.mojo
```

All should compile and run without errors.

---

## Verification Checklist

After each step, verify compilation:

- [ ] Step 1: extensor.mojo compiles
- [ ] Step 2: conftest.mojo compiles
- [ ] Step 3: elementwise.mojo compiles
- [ ] Step 4: activation.mojo compiles
- [ ] Step 5: test_tensors.mojo compiles
- [ ] Step 6: test_activations.mojo compiles
- [ ] Step 7: test_layers.mojo compiles
- [ ] Step 7: test_advanced_activations.mojo compiles

---

## Common Issues

### Issue: "ExTensor still not ImplicitlyCopyable"
**Cause**: Trait definition error
**Fix**: Ensure line 43 has: `struct ExTensor(Copyable, Movable, ImplicitlyCopyable):`

### Issue: "assert_dtype_equal not found"
**Cause**: Function not added to conftest.mojo
**Fix**: Add the function definition from Step 2

### Issue: "exp() still failing"
**Cause**: Type annotation incomplete
**Fix**: Use exact code from Step 4 with `: ExTensor` type hints

### Issue: "test_tensors.mojo still failing on assert_equal"
**Cause**: Forgot to update calls
**Fix**: Replace all 17 instances of `assert_equal(x.dtype(), ...)` with `assert_dtype_equal(...)`

---

## Time Breakdown

| Step | Task | Time |
|------|------|------|
| 1 | Fix ExTensor init | 2 min |
| 2 | Add test helpers | 5 min |
| 3 | Fix scalar abs() | 3 min |
| 4 | Fix exp() type hints | 5 min |
| 5 | Update test_tensors | 10 min |
| 6 | Fix test_activations | 2 min |
| 7 | Verify all tests | 5 min |
| **TOTAL** | | **32 min** |

---

## Next Steps After Fixes

1. Commit changes with issue reference
2. Run full test suite: `pixi run pytest tests/shared/core/`
3. Update CI/CD configuration if needed
4. Address legacy test imports (separate issue)
5. Document any additional fixes needed

---

## Reference Documents

- **Full Analysis**: [README.md](./README.md)
- **Error Catalog**: [ERROR_CATALOG.md](./ERROR_CATALOG.md)
- **Source Files**:
  - extensor.mojo: `/home/mvillmow/ml-odyssey/shared/core/extensor.mojo`
  - conftest.mojo: `/home/mvillmow/ml-odyssey/tests/shared/conftest.mojo`
  - elementwise.mojo: `/home/mvillmow/ml-odyssey/shared/core/elementwise.mojo`
  - activation.mojo: `/home/mvillmow/ml-odyssey/shared/core/activation.mojo`

---

## Questions?

Refer to:
- [ERROR_CATALOG.md](./ERROR_CATALOG.md) for detailed error explanations
- [README.md](./README.md) for comprehensive analysis
- Mojo documentation: https://docs.modular.com/mojo/manual/types

