# Complete Dtype Refactoring Strategy - All Modules

**Date:** 2025-01-20
**Scope:** Refactor all arithmetic, elementwise, and remaining activation functions

---

## Current Status Analysis

### activation.mojo ✅ PARTIALLY COMPLETE

- **File size:** 1,244 lines
- **Refactored:** 6 functions (relu, sigmoid, tanh + backwards)
- **Pattern:** Direct dtype branching with bitcast
- **Reduction achieved:** 133 lines (80% per function)

### elementwise.mojo ⏳ NEEDS REFACTORING

- **File size:** 817 lines
- **Functions:** ~26 operations
- **Current pattern:** Uses `_get_float64()/_set_float64()` conversion
- **Problem:** Converts to/from float64 on every access (inefficient)
- **Refactoring approach:** Replace conversion with dtype dispatch

### arithmetic.mojo ⏳ NEEDS REFACTORING

- **File size:** 734 lines
- **Functions:** ~12 operations
- **Current pattern:** Uses `_get_float64()/_set_float64()` conversion
- **Problem:** Same as elementwise - conversion overhead
- **Refactoring approach:** Replace conversion with dtype dispatch

---

## Refactoring Strategy

### Option 1: Keep Current Pattern (NO - Inefficient)

Current elementwise/arithmetic pattern:

```mojo
for i in range(numel):
    let val = tensor._get_float64(i)  # Conversion overhead!
    result._set_float64(i, operation(val))  # Conversion overhead!
```text

**Problem:** Double conversion (dtype→float64→dtype) on every element access.

### Option 2: Dtype Dispatch (YES - Efficient)

New pattern matching activation.mojo:

```mojo
fn _abs_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    @parameter
    if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_abs(Float32(x)))
    else:
        return Scalar[T](math_abs(Float64(x)))

fn abs(tensor: ExTensor) raises -> ExTensor:
    return dispatch_unary[_abs_op](tensor)
```text

### Benefits:

- Zero conversion overhead
- Type-safe at compile time
- Consistent pattern across all modules

---

## Implementation Plan

### Phase 1: Elementwise Unary Operations (HIGH PRIORITY)

### Functions to refactor (13 unary):

1. abs - absolute value
1. sign - sign function (-1, 0, 1)
1. exp - exponential
1. log - natural logarithm
1. sqrt - square root
1. sin - sine
1. cos - cosine
1. tanh - hyperbolic tangent (duplicate from activation)
1. ceil - ceiling
1. floor - floor
1. round - round to nearest
1. trunc - truncate
1. logical_not - logical not

### Pattern:

```mojo
fn _operation_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    # Use math library with proper dtype casting
    return Scalar[T](math_operation(Float64(x)))

fn operation(tensor: ExTensor) raises -> ExTensor:
    return dispatch_unary[_operation_op](tensor)
```text

**Expected reduction:** ~400 lines → ~130 lines (67% reduction)

### Phase 2: Elementwise Binary Operations (MEDIUM PRIORITY)

### Functions to refactor (3 binary):

1. logical_and - element-wise AND
1. logical_or - element-wise OR
1. logical_xor - element-wise XOR

### Pattern:

```mojo
fn _operation_op[T: DType](a: Scalar[T], b: Scalar[T]) -> Scalar[T]:
    # Logical operation
    return ...

fn operation(a: ExTensor, b: ExTensor) raises -> ExTensor:
    return dispatch_binary[_operation_op](a, b)
```text

**Expected reduction:** ~150 lines → ~40 lines (73% reduction)

### Phase 3: Elementwise Backward Passes (HIGH PRIORITY)

### Functions to refactor (5 backward):

1. exp_backward - exp gradient
1. log_backward - log gradient
1. sqrt_backward - sqrt gradient
1. abs_backward - abs gradient
1. clip_backward - clip gradient

### Pattern:

```mojo
fn _operation_backward_op[T: DType](grad: Scalar[T], x_or_y: Scalar[T]) -> Scalar[T]:
    return grad * derivative(x_or_y)

fn operation_backward(grad_output: ExTensor, x_or_output: ExTensor) raises -> ExTensor:
    return dispatch_binary[_operation_backward_op](grad_output, x_or_output)
```text

**Expected reduction:** ~150 lines → ~50 lines (67% reduction)

### Phase 4: Arithmetic Operations (HIGH PRIORITY)

### Functions to refactor (12 operations):

1. add - addition with broadcasting
1. subtract - subtraction with broadcasting
1. multiply - multiplication with broadcasting
1. divide - division with broadcasting
1. floor_divide - floor division
1. modulo - modulo
1. power - power
1. add_backward - addition gradient
1. subtract_backward - subtraction gradient
1. multiply_backward - multiplication gradient
1. divide_backward - division gradient

**Challenge:** These have broadcasting logic that must be preserved.

### Pattern:

```mojo
fn _add_op[T: DType](a: Scalar[T], b: Scalar[T]) -> Scalar[T]:
    return a + b

fn add(a: ExTensor, b: ExTensor) raises -> ExTensor:
    # Keep broadcasting logic
    let result_shape = broadcast_shapes(a.shape(), b.shape())
    # ... broadcasting setup ...

    # Replace operation with dispatch
    for result_idx in range(total_elems):
        # ... compute idx_a, idx_b ...
        result._set_at_dispatch[_add_op](result_idx,
                                          a._get_at(idx_a),
                                          b._get_at(idx_b))
```text

**Note:** May need specialized dispatch helper for broadcasted operations.

**Expected reduction:** ~500 lines → ~250 lines (50% reduction)

### Phase 5: Remaining Activation Functions (LOWER PRIORITY)

### Parametric functions (4):

1. leaky_relu (alpha parameter)
1. prelu (alpha tensor)
1. elu (alpha parameter)
1. gelu (approximate boolean)

### Complex functions (1):

1. softmax (axis-wise reduction)

### Already optimized (2):

1. swish (uses composition)
1. mish (uses composition)

**Expected reduction:** ~200 lines → ~100 lines (50% reduction)

---

## Projected Total Impact

### Code Reduction Summary

| Module         | Current | After  | Removed | Reduction |
|----------------|---------|--------|---------|-----------|
| activation.mojo| 1,244   | 1,100  | 144     | 12%       |
| elementwise.mojo| 817    | 220    | 597     | 73%       |
| arithmetic.mojo| 734     | 384    | 350     | 48%       |
| **TOTAL**      | **2,795**| **1,704**| **1,091**| **39%** |

### Additional Benefits

### Performance Improvements:

- Eliminate dtype conversion overhead in elementwise/arithmetic
- Direct dtype access (no float64 conversion round-trips)
- Estimated 10-30% speedup for elementwise operations

### Code Quality:

- Consistent dispatch pattern across all modules
- Type-safe compile-time specialization
- Single source of truth for all operations

---

## Implementation Order (Priority)

1. **Elementwise unary operations** (highest impact, simplest)
1. **Elementwise backward passes** (gradient correctness critical)
1. **Elementwise binary operations** (logical ops)
1. **Arithmetic operations** (complex due to broadcasting)
1. **Remaining activations** (parametric functions, lower priority)

---

## Technical Considerations

### Broadcasting Compatibility

Arithmetic operations have broadcasting logic that must be preserved:

```mojo
// Before
for result_idx in range(total_elems):
    let a_val = a._get_float64(idx_a)
    let b_val = b._get_float64(idx_b)
    result._set_float64(result_idx, a_val + b_val)

// After (need helper)
for result_idx in range(total_elems):
    result._set_at_binary[_add_op](result_idx,
                                     a, idx_a,
                                     b, idx_b)
```text

**Solution:** May need new dispatch helper for broadcasted element access.

### Math Library Dtype Support

Some math functions only support Float32/Float64:

- `math.exp`, `math.log`, `math.sqrt`, `math.sin`, `math.cos`
- Need dtype casting in operation functions

### Backward Pass Patterns

Backward passes have two patterns:

1. Takes `x` (input) - exp, log, abs
1. Takes `output` (forward result) - sqrt (for efficiency)

Must preserve these patterns for numerical stability.

---

## Next Steps

1. Start with elementwise unary operations (simplest, highest impact)
1. Validate with test suite after each module
1. Measure performance improvements (eliminate conversion overhead)
1. Document pattern for future contributors
1. Consider creating specialized broadcast dispatcher if needed

---

## Success Criteria

- [x] Dtype dispatch infrastructure created
- [x] 6 activation functions refactored (80% reduction)
- [ ] 13 elementwise unary operations refactored (67% reduction expected)
- [ ] 5 elementwise backward passes refactored (67% reduction expected)
- [ ] 3 elementwise binary operations refactored (73% reduction expected)
- [ ] 12 arithmetic operations refactored (50% reduction expected)
- [ ] All tests pass with no regressions
- [ ] Performance maintained or improved
- [ ] Total reduction: 1,091+ lines across all modules

---

**Ready to begin comprehensive refactoring of all modules!**
