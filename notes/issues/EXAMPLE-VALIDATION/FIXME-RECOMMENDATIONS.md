# FIXME Recommendations - Example Validation

## Priority 1: Framework-Level Fixes (Must fix first - blocks all examples)

### Fix 1.1: Update ExTensor Struct Conformances

**File**: `/home/mvillmow/ml-odyssey/shared/core/types/extensor.mojo`

**Issue**: `ExTensor` doesn't implement `Copyable` and `Movable` traits, causing type errors when used in collections or returned from functions.

**Error Pattern**:
```
error: value of type 'ExTensor' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'
error: cannot bind type 'ExTensor' to trait 'Copyable & Movable'
```

**Required Changes**:

```mojo
# Current (broken):
struct ExTensor:
    var data: Pointer[Float64]
    var shape: DynamicVector[Int]
    var strides: DynamicVector[Int]
    var dtype: DType

# Fixed (needed):
@fieldwise_init
struct ExTensor:
    var data: Pointer[Float64]
    var shape: DynamicVector[Int]
    var strides: DynamicVector[Int]
    var dtype: DType

    # Add Copyable conformance
    fn __copyinit__(self) -> Self:
        """Create independent copy of tensor."""
        # Deep copy data, shape, strides
        return Self(
            data=copy_pointer_data(self.data, self.num_elements()),
            shape=DynamicVector[Int](self.shape),
            strides=DynamicVector[Int](self.strides),
            dtype=self.dtype
        )

    # Add Movable conformance
    fn __moveinit__(inout self, owned other: Self):
        """Take ownership of tensor resources."""
        self.data = other.data
        self.shape = other.shape
        self.strides = other.strides
        self.dtype = other.dtype

    # Add copy() method for explicit copying
    fn copy(self) -> Self:
        """Return independent copy of tensor."""
        return Self(self)
```

**Files Affected**:
- All files importing ExTensor
- All examples using ExTensor

**Validation**:
```bash
# After fix, verify:
pixi run mojo build shared/core/types/extensor.mojo
# Should compile without trait conformance errors
```

---

### Fix 1.2: Add Missing ExTensor Methods

**File**: `/home/mvillmow/ml-odyssey/shared/core/types/extensor.mojo`

**Issue**: Examples call methods that don't exist: `ExTensor.full()`, `tensor.copy()`

**Required Methods**:

```mojo
struct ExTensor:
    # ... existing code ...

    # Static factory method
    fn full[dtype: DType](shape: DynamicVector[Int], value: Float64) -> Self:
        """Create tensor filled with value."""
        # Allocate and fill
        # Return ExTensor with initialized data

    # Instance method for copying
    fn copy(self) -> Self:
        """Return independent copy of this tensor."""
        # Already added in Fix 1.1
        return Self(self)

    # In-place operations that use 'inout' (not 'inout self')
    fn normalize_(inout self):
        """Normalize tensor in-place."""
        # Implementation

    fn clamp_(inout self, min_val: Float64, max_val: Float64):
        """Clamp values in-place."""
        # Implementation
```

**Examples Needing This**:
- mixed_precision_training.mojo (needs `ExTensor.full()`)
- All autograd examples (need tensor manipulation methods)

---

### Fix 1.3: Fix Module Exports in shared/core/__init__.mojo

**File**: `/home/mvillmow/ml-odyssey/shared/core/__init__.mojo`

**Current Issues**:
- `Tensor` type not exported (examples can't import it)
- `Module` and `Linear` classes not exported
- `creation` module functions not accessible

**Required Changes**:

```mojo
# File: shared/core/__init__.mojo

# Add proper exports
from shared.core.types import ExTensor as Tensor  # Alias for compatibility
from shared.core.types import (
    ExTensor,
    DType,
    # ... other types
)

from shared.core.layers import (
    Module,  # Ensure this exists or create it
    Linear,  # Ensure this exists or create it
)

from shared.core.creation import (
    zeros,
    ones,
    full,
    arange,
    eye,
    linspace,
)

# Make sure __all__ is valid Mojo syntax (no file-scope lists in Mojo 0.25+)
# Option 1: Remove __all__ entirely (import statements define exports)
# Option 2: If needed, use module-level exports

# DO NOT USE (this causes syntax errors):
# __all__ = ["Tensor", "ExTensor", ...]  # WRONG - file scope expressions not allowed
```

**Files Affected**:
- attention_layer.mojo (imports Module, Linear)
- prelu_activation.mojo (imports Module)
- trait_example.mojo (imports Tensor)
- All custom layer examples

---

### Fix 1.4: Fix Data Type Decorators

**Files**:
- `/shared/core/types/fp8.mojo`
- `/shared/core/types/bf8.mojo`
- `/shared/core/types/integer.mojo`
- `/shared/core/types/unsigned.mojo`

**Issue**: `@value` decorator removed in recent Mojo versions

**Required Changes**:

```mojo
# Current (broken):
@value
struct FP8:
    var value: UInt8

    fn __init__(inout self, value: UInt8 = 0):
        self.value = value

# Fixed:
@fieldwise_init
struct FP8:
    var value: UInt8

    fn __init__(self, value: UInt8 = 0) -> Self:
        return Self(value=value)

    # Explicit conformances
    fn __copyinit__(self) -> Self:
        return Self(value=self.value)

    fn __moveinit__(inout self, owned other: Self):
        self.value = other.value
```

**Pattern for All Type Definitions**:
1. Replace `@value` with `@fieldwise_init`
2. Update `__init__` to return `Self` (not use `inout self`)
3. Add `__copyinit__` for Copy trait
4. Add `__moveinit__` for Move trait
5. Replace string conversion: `str(value)` → `String(value)` or `repr(value)`

**Validation After Fix**:
```bash
pixi run mojo run -I . examples/fp8_example.mojo 2>&1 | head -20
# Should reach string conversion errors, not decorator errors
```

---

### Fix 1.5: Fix Module Path Issues

**Issues to Resolve**:

1. **`collections.vector` (DynamicVector not found)**
   - Search for correct location of DynamicVector
   - Update imports in affected files
   - Or: Replace with built-in collections if available

2. **`sys.info.simdwidthof` (missing function)**
   - Check Mojo standard library documentation
   - Find replacement or alternative
   - Update performance examples

3. **`shared.core.creation` module**
   - Ensure module exists and exports: `zeros`, `ones`, `full`, `arange`, `eye`, `linspace`
   - Create if missing
   - Update imports accordingly

4. **`src.extensor` path**
   - Completely wrong path
   - Should be: `from shared.core import ExTensor`
   - Affects: basic_usage.mojo, test_arithmetic.mojo

**Search Commands**:
```bash
# Find DynamicVector
find /home/mvillmow/ml-odyssey -name "*.mojo" -type f -exec grep -l "DynamicVector" {} \;

# Find simdwidthof
find /home/mvillmow/ml-odyssey -name "*.mojo" -type f -exec grep -l "simdwidthof" {} \;

# Find creation module functions
grep -r "def zeros" /home/mvillmow/ml-odyssey/shared/
grep -r "fn zeros" /home/mvillmow/ml-odyssey/shared/
```

---

## Priority 2: Syntax Modernization (Fix after framework issues)

### Fix 2.1: Remove `inout self` from Method Definitions

**Pattern**:
```mojo
# WRONG (Mojo 0.24.1+):
fn forward(inout self, borrowed input: Tensor) -> Tensor:
    self.do_something()
    return result

# CORRECT:
fn forward(self, borrowed input: Tensor) -> Tensor:
    self.do_something()
    return result
```

**Files Affected** (20+ occurrences):
- trait_example.mojo
- ownership_example.mojo
- simd_example.mojo
- performance examples
- custom layer examples
- trait_based_layer.mojo
- All layer __init__ methods

**Automated Fix**:
```bash
# Replace pattern across files
cd /home/mvillmow/ml-odyssey
find examples -name "*.mojo" -type f -exec sed -i 's/fn \([a-zA-Z_][a-zA-Z0-9_]*\)(inout self,/fn \1(self,/g' {} \;
find examples -name "*.mojo" -type f -exec sed -i 's/fn \([a-zA-Z_][a-zA-Z0-9_]*\)(inout self)/fn \1(self)/g' {} \;

# Verify changes
git diff examples/ | head -100
```

---

### Fix 2.2: Remove `owned` Parameter Deprecation

**Pattern**:
```mojo
# OLD (deprecated):
fn consume_tensor(owned tensor: Tensor) -> Float64:
    return tensor.sum()

# NEW (use move semantics with ^):
fn consume_tensor(tensor: ExTensor) -> Float64:
    # Ownership is implicitly taken in Mojo 0.24.1+
    return tensor.sum()
```

**Note**: Some functions may need to use `^` when transferring ownership:
```mojo
fn create_and_move() -> ExTensor:
    var tensor = ExTensor.zeros([10, 10])
    return tensor^  # Move ownership to caller
```

---

### Fix 2.3: Fix String Conversion Issues

**Pattern - Problem Areas**:

1. **fp8.mojo, bf8.mojo, integer.mojo** - Using `str()` function

```mojo
# WRONG:
return "FP8(" + str(self.to_float32()) + ")"

# CORRECT (option 1 - use String constructor):
return String("FP8(") + String(self.to_float32()) + ")"

# CORRECT (option 2 - if String.__add__ supports this):
return "FP8(" + String(self.to_float32())[...] + ")"
```

2. **integer_example.mojo** - Multiple `str()` calls

```mojo
# WRONG:
print("Int8 values:", str(i8_1), str(i8_2))

# CORRECT:
print("Int8 values:", String(i8_1), String(i8_2))
# Or if type has __str__ method, use directly in print
print("Int8 values:", i8_1, i8_2)
```

---

### Fix 2.4: Fix Deprecated Language Features

**Issue: `let` keyword in autograd/functional.mojo**

```mojo
# WRONG (Python, not Mojo):
let val = tensor._get_float64(i)

# CORRECT:
var val = tensor._get_float64(i)
```

**Issue: `__all__` at file scope in shared/autograd/__init__.mojo**

```mojo
# WRONG (file-scope expressions not allowed):
__all__ = [
    "ExTensor",
    "Variable",
    # ...
]

# CORRECT (option 1 - remove, let imports define exports):
# Delete __all__ entirely

# CORRECT (option 2 - if Mojo 0.24.1 supports it):
# Check Mojo documentation for proper syntax
```

---

## Priority 3: Module-Level Fixes (Create missing modules)

### Fix 3.1: Create Missing `shared/core/creation.mojo`

**If module doesn't exist**, create it with tensor factory functions:

```mojo
# File: shared/core/creation.mojo

from shared.core.types import ExTensor, DType

fn zeros(shape: DynamicVector[Int], dtype: DType = DType.float64) -> ExTensor:
    """Create tensor filled with zeros."""
    # Allocate memory
    # Fill with 0.0
    # Return ExTensor

fn ones(shape: DynamicVector[Int], dtype: DType = DType.float64) -> ExTensor:
    """Create tensor filled with ones."""
    # Similar to zeros but fill with 1.0

fn full(shape: DynamicVector[Int], value: Float64, dtype: DType = DType.float64) -> ExTensor:
    """Create tensor filled with specific value."""

fn arange(start: Float64, stop: Float64, step: Float64 = 1.0) -> ExTensor:
    """Create 1D tensor with evenly spaced values."""

fn eye(n: Int, dtype: DType = DType.float64) -> ExTensor:
    """Create identity matrix."""

fn linspace(start: Float64, stop: Float64, num: Int) -> ExTensor:
    """Create 1D tensor with linearly spaced values."""
```

---

### Fix 3.2: Create/Update Module Base Class

**File**: `/shared/core/layers.mojo` or `/shared/core/module.mojo`

**Required Interface**:

```mojo
trait Module:
    """Base class for neural network layers."""

    fn forward(self, input: ExTensor) -> ExTensor:
        """Forward pass."""
        raise "Not implemented"

    fn parameters(self) -> List[ExTensor]:
        """Return layer parameters."""
        raise "Not implemented"

    fn zero_grad(inout self):
        """Zero out gradients."""
        raise "Not implemented"

struct Linear(Module):
    """Fully connected layer."""
    var weights: ExTensor
    var bias: ExTensor

    fn __init__(self, input_size: Int, output_size: Int) -> Self:
        # Initialize weights and bias
        return Self(weights=..., bias=...)

    fn forward(self, input: ExTensor) -> ExTensor:
        # x @ W.T + b
        return ...

    fn parameters(self) -> List[ExTensor]:
        return List[ExTensor]()  # Return copy of weights/bias

    fn zero_grad(inout self):
        # Zero bias grad
        pass
```

---

## Testing & Validation Plan

### Step 1: Verify Framework Fixes

```bash
# After fixing ExTensor struct
pixi run mojo build shared/core/types/extensor.mojo
# Should succeed

# After fixing module exports
pixi run mojo build shared/core/__init__.mojo
# Should succeed
```

### Step 2: Verify Individual Fixes

```bash
# After fixing fp8.mojo
pixi run mojo run -I . examples/fp8_example.mojo 2>&1

# After fixing trait_example.mojo
pixi run mojo run -I . examples/mojo-patterns/trait_example.mojo 2>&1

# After fixing basic_usage.mojo
pixi run mojo run -I . examples/basic_usage.mojo 2>&1
```

### Step 3: Full Validation

```bash
# Run all examples
for example in examples/**/*.mojo; do
    echo "Testing: $example"
    pixi run mojo run -I . "$example" 2>&1 | head -5
done
```

---

## Files Requiring Changes

### High Priority (Framework - must fix first)

1. **shared/core/types/extensor.mojo**
   - Add Copyable/Movable traits
   - Add copy() method
   - Add full() static method

2. **shared/core/types/fp8.mojo**
   - Replace @value with @fieldwise_init
   - Fix __init__ signature
   - Fix string conversion

3. **shared/core/types/bf8.mojo**
   - Same as fp8.mojo

4. **shared/core/types/integer.mojo**
   - Replace @value with @fieldwise_init
   - Fix __init__ signature
   - Fix string conversion

5. **shared/core/types/unsigned.mojo**
   - Same as integer.mojo

6. **shared/core/__init__.mojo**
   - Fix module exports
   - Remove __all__ file-scope list
   - Export Tensor, Module, Linear, creation functions

7. **shared/autograd/__init__.mojo**
   - Remove __all__ file-scope list

### Medium Priority (Syntax - fix after framework)

8. **All files in examples/**
   - Remove `inout self` from methods
   - Fix `owned` parameters
   - Fix string conversion
   - Fix module imports

9. **shared/autograd/functional.mojo**
   - Replace `let` with `var`
   - Fix move semantics

10. **shared/training/mixed_precision.mojo**
    - Fix `inout` parameters
    - Fix move semantics

---

## Estimated Effort

| Priority | Category | Files | Est. Hours |
|----------|----------|-------|-----------|
| 1 | Framework - Type System | 6 | 4-6 |
| 1 | Framework - Exports | 2 | 2-3 |
| 2 | Syntax - inout self | 20+ | 2-3 |
| 2 | Syntax - String conversion | 4 | 1-2 |
| 2 | Syntax - Keywords | 2 | 1-2 |
| 3 | Module Creation | 2 | 4-6 |
| **Total** | | **38+** | **14-22 hours** |

---

## Follow-up Actions

After fixes are complete:

1. **Add Example CI Validation**
   - Create workflow to compile/run examples on every PR
   - Catch regressions early

2. **Update Documentation**
   - Update example README with latest Mojo syntax
   - Add migration guide for Mojo 0.24.1+ changes

3. **Create Regression Tests**
   - Test each example file compiles
   - Test key functionality runs
   - Prevent future breakage

4. **Review Mojo Version Compatibility**
   - Document minimum required Mojo version
   - Update pixi.toml with correct version constraint
