# Test Execution Fixes - Code Snippets

**Ready-to-use code patches for all identified errors**

---

## Critical Fixes (Compilation Blockers)

### Fix #1: ExTensor Constructor Parameters

**File**: `/shared/core/extensor.mojo`

**Lines**: 89, 149

#### Change 1: **init** method (line 89)

```mojo
// BEFORE:
fn __init__(mut self, shape: List[Int], dtype: DType) raises:
    """Initialize a new ExTensor with given shape and dtype.

    Args:
        `shape`: The shape of the tensor as a vector of dimension sizes.
        `dtype`: The data type of tensor elements.

    Raises:
        Error: If tensor size exceeds MAX_TENSOR_BYTES (2 GB)

    Note:
        This is a low-level constructor. Users should prefer creation
        functions like zeros(), ones(), full(), etc.
    """

// AFTER:
fn __init__(out self, shape: List[Int], dtype: DType) raises:
    """Initialize a new ExTensor with given shape and dtype.

    Args:
        `shape`: The shape of the tensor as a vector of dimension sizes.
        `dtype`: The data type of tensor elements.

    Raises:
        Error: If tensor size exceeds MAX_TENSOR_BYTES (2 GB)

    Note:
        This is a low-level constructor. Users should prefer creation
        functions like zeros(), ones(), full(), etc.
    """
```

#### Change 2: **copyinit** method (line 149)

```mojo
// BEFORE:
fn __copyinit__(mut self, existing: Self):
    """Copy constructor - creates shared ownership with reference counting.

    Creates a new reference to the same underlying data.
    Increments the reference count to track shared ownership.
    This prevents double-free and enables safe view semantics.

    Fixes: #1908 (MOJO-005), part of #1906 (MOJO-003)
    """

// AFTER:
fn __copyinit__(out self, existing: Self):
    """Copy constructor - creates shared ownership with reference counting.

    Creates a new reference to the same underlying data.
    Increments the reference count to track shared ownership.
    This prevents double-free and enables safe view semantics.

    Fixes: #1908 (MOJO-005), part of #1906 (MOJO-003)
    """
```

---

### Fix #2: Metric Base Class Constructors

**File**: `/shared/training/metrics/base.mojo`

**Lines**: 59, 220

#### Change 1: MetricResult.**init** scalar (line 59)

```mojo
// BEFORE:
fn __init__(mut self, name: String, value: Float64):
    """Create scalar metric result."""
    self.name = name
    self.is_scalar = True
    self.scalar_value = value
    self.tensor_value = ExTensor(List[Int]())  # Placeholder

// AFTER:
fn __init__(out self, name: String, value: Float64):
    """Create scalar metric result."""
    self.name = name
    self.is_scalar = True
    self.scalar_value = value
    self.tensor_value = ExTensor(List[Int]())  # Placeholder
```

#### Change 2: MetricBase.**init** (line 220)

```mojo
// BEFORE:
fn __init__(mut self):
    """Initialize metric with empty state."""
    pass

// AFTER:
fn __init__(out self):
    """Initialize metric with empty state."""
    pass
```

---

### Fix #3: Accuracy Metric Constructor

**File**: `/shared/training/metrics/accuracy.mojo`

**Line**: 377

```mojo
// BEFORE:
fn __init__(mut self):
    """Initialize accuracy tracker."""
    self._correct = 0
    self._total = 0

// AFTER:
fn __init__(out self):
    """Initialize accuracy tracker."""
    self._correct = 0
    self._total = 0
```

---

### Fix #4: Loss Tracker Constructor

**File**: `/shared/training/metrics/loss_tracker.mojo`

**Line**: 231

```mojo
// BEFORE:
fn __init__(mut self, window_size: Int = 100):
    """Initialize loss tracker with given window size."""
    self._window_size = window_size
    self._losses = List[Float64]()

// AFTER:
fn __init__(out self, window_size: Int = 100):
    """Initialize loss tracker with given window size."""
    self._window_size = window_size
    self._losses = List[Float64]()
```

---

### Fix #5: Confusion Matrix Constructor

**File**: `/shared/training/metrics/confusion_matrix.mojo`

**Line**: 58

```mojo
// BEFORE:
fn __init__(mut self, num_classes: Int, class_names: List[String] = List[String]()) raises:
    """Initialize confusion matrix."""
    self._num_classes = num_classes
    self._class_names = class_names
    # ... rest of init

// AFTER:
fn __init__(out self, num_classes: Int, class_names: List[String] = List[String]()) raises:
    """Initialize confusion matrix."""
    self._num_classes = num_classes
    self._class_names = class_names
    # ... rest of init
```

---

### Fix #6: FP8 Type Constructor

**File**: `/shared/core/types/fp8.mojo`

**Lines**: 36, 87

#### Change 1: FP8.**init** (line 36)

```mojo
// BEFORE:
fn __init__(mut self, value: UInt8 = 0):
    """Initialize FP8 value."""
    self.value = value

// AFTER:
fn __init__(out self, value: UInt8 = 0):
    """Initialize FP8 value."""
    self.value = value
```

#### Change 2: int() → Int() (line 87)

```mojo
// BEFORE:
fn to_fp8(x: Float32) -> FP8:
    """Convert Float32 to FP8."""
    var sign = 1 if x >= 0 else -1
    var abs_x = abs(x) if x >= 0 else -x
    var mantissa = int(abs_x * 128.0)  # Scale to 3-bit range
    var encoded_value = UInt8((sign_bit << 7) | (mantissa & 0x7F))
    return FP8(encoded_value)

// AFTER:
fn to_fp8(x: Float32) -> FP8:
    """Convert Float32 to FP8."""
    var sign = 1 if x >= 0 else -1
    var abs_x = abs(x) if x >= 0 else -x
    var mantissa = Int(abs_x * 128.0)  # Scale to 3-bit range
    var encoded_value = UInt8((sign_bit << 7) | (mantissa & 0x7F))
    return FP8(encoded_value)
```

---

### Fix #7: Integer Type Constructor

**File**: `/shared/core/types/integer.mojo`

**Line**: 39

```mojo
// BEFORE:
fn __init__(mut self, value: Int8):
    """Initialize Int8 wrapper."""
    self.value = value

// AFTER:
fn __init__(out self, value: Int8):
    """Initialize Int8 wrapper."""
    self.value = value
```

---

### Fix #8: MXFP4 Type Constructor

**File**: `/shared/core/types/mxfp4.mojo`

**Lines**: 57, 103, 474

#### Change 1: MXFP4.**init** (line 57)

```mojo
// BEFORE:
fn __init__(mut self, exponent: UInt8 = 127):
    """Initialize MXFP4 with shared exponent."""
    self.exponent = exponent

// AFTER:
fn __init__(out self, exponent: UInt8 = 127):
    """Initialize MXFP4 with shared exponent."""
    self.exponent = exponent
```

#### Change 2: Type casting API (line 103)

```mojo
// BEFORE:
fn get_scale(self) -> E8M0Scale:
    """Get scale value."""
    var biased_exp = self.exponent
    return E8M0Scale(biased_exp.cast[DType.uint8]())

// AFTER:
fn get_scale(self) -> E8M0Scale:
    """Get scale value."""
    var biased_exp = self.exponent
    return E8M0Scale(UInt8(biased_exp))
```

#### Change 3: E8M0Scale.**init** (line 474)

```mojo
// BEFORE:
fn __init__(mut self):
    """Initialize scale value."""
    self.value = 1

// AFTER:
fn __init__(out self):
    """Initialize scale value."""
    self.value = 1
```

---

### Fix #9: NVFP4 Type Constructor

**File**: `/shared/core/types/nvfp4.mojo`

**Lines**: 65, 94, 525

#### Change 1: NVFP4.**init** (line 65)

```mojo
// BEFORE:
fn __init__(mut self, value: UInt8 = 0x38):
    """Initialize NVFP4 value."""
    self.value = value

// AFTER:
fn __init__(out self, value: UInt8 = 0x38):
    """Initialize NVFP4 value."""
    self.value = value
```

#### Change 2: int() → Int() (line 94)

```mojo
// BEFORE:
fn to_nvfp4(x: Float32, scale: Float32) -> NVFP4:
    """Convert Float32 to NVFP4 with scale."""
    var mantissa = int(scale * 512.0)  # Scale to mantissa range
    var encoded = UInt8(mantissa & 0xFF)
    return NVFP4(encoded)

// AFTER:
fn to_nvfp4(x: Float32, scale: Float32) -> NVFP4:
    """Convert Float32 to NVFP4 with scale."""
    var mantissa = Int(scale * 512.0)  # Scale to mantissa range
    var encoded = UInt8(mantissa & 0xFF)
    return NVFP4(encoded)
```

#### Change 3: E4M3Scale.**init** (line 525)

```mojo
// BEFORE:
fn __init__(mut self):
    """Initialize scale value."""
    self.scale_factor = 1.0

// AFTER:
fn __init__(out self):
    """Initialize scale value."""
    self.scale_factor = 1.0
```

---

## High Priority Fixes (Test-Specific)

### Fix #10: List Mutation Pattern in Integration Test

**File**: `/tests/integration/test_all_architectures.mojo`

**Line**: 49

```mojo
// BEFORE:
try:
    # Create dummy input (batch_size, 3, 32, 32)
    print("Creating dummy input: (" + String(batch_size) + ", 3, 32, 32)")
    var input = zeros(
        List[Int]()
            .append(batch_size)
            .append(3)
            .append(32)
            .append(32),
        DType.float32
    )

// AFTER:
try:
    # Create dummy input (batch_size, 3, 32, 32)
    print("Creating dummy input: (" + String(batch_size) + ", 3, 32, 32)")
    var shape = List[Int]()
    shape.append(batch_size)
    shape.append(3)
    shape.append(32)
    shape.append(32)
    var input = zeros(shape, DType.float32)
```

---

### Fix #11: Test Assertions (Data Integrity Test)

**File**: `/tests/test_data_integrity.mojo`

**Multiple lines**: 30, 51, 82, 107, 129, 190, 204, 233, 254

#### Option A: Update to testing module (recommended)

```mojo
// AT TOP OF FILE:
from testing import assert_true, assert_false, assert_equal

// Then replace all assert statements:
// BEFORE:
assert decoded.numel() == 32, "Decoded size should be 32"

// AFTER:
assert_equal(decoded.numel(), 32)
```

#### Option B: Use conditional checks

```mojo
// BEFORE:
assert decoded.numel() == 32, "Decoded size should be 32"

// AFTER:
if not (decoded.numel() == 32):
    raise Error("Decoded size should be 32")
```

**All instances to fix**:

- Line 30: `assert decoded.numel() == 32, "Decoded size should be 32"`
- Line 51: `assert (encoded.numel() == 32), "Encoded size"`
- Line 82: `assert (t_fp8.numel() == 10), "FP8 tensor should have 10 elements"`
- Line 107: `assert (t_int8.numel() == 10), "Int8 tensor should have 10 elements"`
- Line 129: `assert fp8_t.numel() == 10, "FP8 tensor should have 10 elements"`
- Line 190: `assert decoded.numel() == 10, "Decoded should have 10 elements"`
- Line 204: `assert i8_t.numel() == 10, "Int8 tensor should have 10 elements"`
- Line 233: `assert (compressed.numel() < original.numel()), "Compressed should be smaller"`
- Line 254: `assert (decompressed.numel() == original.numel()), "Decompressed size matches"`

---

### Fix #12: Float Type Conversions

**File**: `/tests/test_data_integrity.mojo`

**Lines**: 181, 167

#### Change 1: Float32 to Float16 conversion (line 181)

```mojo
// BEFORE:
fn test_float16_conversion():
    """Test Float16 tensor conversion."""
    var t_fp16 = ExTensor(List[Int](10), DType.float16)
    for i in range(10):
        var val_f32 = Float32(i)
        t_fp16._data.bitcast[Float16]()[i] = val_f32  # ERROR: implicit conversion

// AFTER:
fn test_float16_conversion():
    """Test Float16 tensor conversion."""
    var t_fp16 = ExTensor(List[Int](10), DType.float16)
    for i in range(10):
        var val_f32 = Float32(i)
        var val_f16 = Float16(val_f32)  # Explicit conversion
        t_fp16._data.bitcast[Float16]()[i] = val_f16
```

#### Change 2: Float64 to Float32 conversion (line 167)

```mojo
// BEFORE:
fn fp4_to_float(encoded: UInt8, scale: Float32) -> Float32:
    """Decode FP4 to Float32."""
    var sign = 1.0 if (encoded & 0x80) == 0 else -1.0
    return 0.0 if sign == 0 else -0.0  # ERROR: ambiguous type

// AFTER:
fn fp4_to_float(encoded: UInt8, scale: Float32) -> Float32:
    """Decode FP4 to Float32."""
    var sign = 1.0 if (encoded & 0x80) == 0 else -1.0
    return Float32(0.0) if sign == 0 else Float32(-0.0)  # Explicit type
```

---

## Medium Priority Fixes (Test Infrastructure)

### Fix #13: Add main() Function to Utility Tests

Add this to each of the 6 utility test files:

**File pattern**: `/tests/shared/utils/test_*.mojo`

```mojo
fn main():
    """Run all configuration tests."""
    print("=" * 70)
    print("Running configuration tests...")
    print("=" * 70)

    # Run all test functions
    try:
        test_load_yaml_config()
        print("✓ test_load_yaml_config")
    except:
        print("✗ test_load_yaml_config")

    try:
        test_load_json_config()
        print("✓ test_load_json_config")
    except:
        print("✗ test_load_json_config")

    # ... add all other test functions

    print("=" * 70)
    print("Tests completed!")
    print("=" * 70)
```

**Apply to**:

- `/tests/shared/utils/test_config.mojo` - Add main() calling all test_* functions
- `/tests/shared/utils/test_logging.mojo` - Add main() calling all test_* functions
- `/tests/shared/utils/test_io.mojo` - Add main() calling all test_* functions
- `/tests/shared/utils/test_profiling.mojo` - Add main() calling all test_* functions
- `/tests/shared/utils/test_random.mojo` - Add main() calling all test_* functions
- `/tests/shared/utils/test_visualization.mojo` - Add main() calling all test_* functions

---

## Patch Application Order

### Recommended Sequence

**Step 1: Apply all constructor fixes (5 minutes)**

1. Fix ExTensor (2 changes)
2. Fix metrics base (2 changes)
3. Fix accuracy (1 change)
4. Fix loss_tracker (1 change)
5. Fix confusion_matrix (1 change)
6. Fix FP8 (1 change)
7. Fix Int8 (1 change)
8. Fix MXFP4 (2 changes)
9. Fix NVFP4 (2 changes)

Expected outcome: Library compiles successfully

**Step 2: Apply test-specific fixes (10 minutes)**

1. Fix integration test List pattern (1 change)
2. Update data integrity assertions (9+ changes)
3. Fix float conversions (2 changes)

Expected outcome: All tests compile and run

**Step 3: Fix test infrastructure (5 minutes)**

1. Add main() to utility tests (6 files)

Expected outcome: All test suites executable

---

## Verification Commands

After applying fixes, verify with:

```bash
# Test compilation individually
pixi run mojo -I . tests/integration/test_all_architectures.mojo
pixi run mojo -I . tests/test_core_operations.mojo
pixi run mojo -I . tests/test_data_integrity.mojo
pixi run mojo -I . tests/shared/utils/test_config.mojo

# All test suites
for file in tests/test_*.mojo tests/integration/test_*.mojo tests/shared/utils/test_*.mojo; do
    echo "Testing: $file"
    pixi run mojo -I . "$file" 2>&1 | grep -E "(error|warning|passed|failed)"
done
```

---

## Summary

**Total changes needed**: 29 (mostly mechanical)
**Files to modify**: 11
**Estimated time**: 40-50 minutes
**Expected result**: All tests compile and executable
