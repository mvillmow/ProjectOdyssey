# Test Execution Error Matrix

**Visual Reference for Error Patterns and Affected Files**

---

## Error Category Heat Map

```
Error Type                          | Severity | Count | Files Affected
====================================|==========|=======|==================
Constructor param (mut → out)       | CRITICAL | 10+   | 6 files
Assertion syntax issues             | MEDIUM   | 12+   | 1 file
Deprecated int() builtin            | CRITICAL | 2     | 2 files
Type conversion issues              | CRITICAL | 2     | 2 files
List mutation pattern               | CRITICAL | 1     | 1 file
Test framework missing              | MEDIUM   | 6     | 6 files
Type casting API changes            | CRITICAL | 1     | 1 file
```

---

## Dependency Chain Visualization

```
Integration Tests (test_all_architectures.mojo)
├── shared/core/__init__.mojo
│   └── shared/core/extensor.mojo (ERROR: __init__ at line 89)
│       └── ExTensor class
│           └── Used by test at line 47-54
│
└── shared/training/metrics/__init__.mojo
    ├── shared/training/metrics/base.mojo (ERROR: __init__ at lines 59, 220)
    ├── shared/training/metrics/accuracy.mojo (ERROR: __init__ at line 377)
    ├── shared/training/metrics/loss_tracker.mojo (ERROR: __init__ at line 231)
    └── shared/training/metrics/confusion_matrix.mojo (ERROR: __init__ at line 58)

Core Operations Test (test_core_operations.mojo)
├── [Same as above]
└── Uses all metrics classes

Data Integrity Test (test_data_integrity.mojo)
├── shared/core/extensor.mojo (ERROR: __init__ at line 89, 149)
├── shared/core/types/fp8.mojo (ERROR: __init__ at line 36, int() at line 87)
├── shared/core/types/integer.mojo (ERROR: __init__ at line 39)
├── shared/core/types/mxfp4.mojo (ERROR: __init__ at line 57, cast at line 103)
└── shared/core/types/nvfp4.mojo (ERROR: __init__ at line 65, int() at line 94)

Utility Tests (6 files)
├── test_config.mojo (ERROR: no main())
├── test_logging.mojo (ERROR: no main())
├── test_io.mojo (ERROR: no main())
├── test_profiling.mojo (ERROR: no main())
├── test_random.mojo (ERROR: no main())
└── test_visualization.mojo (ERROR: no main())
```

---

## File-by-File Error Summary

### CRITICAL ERRORS (Block Compilation)

#### shared/core/extensor.mojo
```
Line 89:  fn __init__(mut self, ...) → Change to: out self
Line 149: fn __copyinit__(mut self, ...) → Change to: out self
Status: 2 changes needed
Impact: Blocks ALL tests that use ExTensor
```

#### shared/training/metrics/base.mojo
```
Line 59:  fn __init__(mut self, name: String, value: Float64) → out self
Line 220: fn __init__(mut self) → out self
Status: 2 changes needed
Impact: Blocks ALL metrics-dependent tests
```

#### shared/training/metrics/accuracy.mojo
```
Line 377: fn __init__(mut self) → out self
Status: 1 change needed
Impact: Blocks accuracy metric tests
```

#### shared/training/metrics/loss_tracker.mojo
```
Line 231: fn __init__(mut self, window_size: Int = 100) → out self
Status: 1 change needed
Impact: Blocks loss tracking tests
```

#### shared/training/metrics/confusion_matrix.mojo
```
Line 58: fn __init__(mut self, num_classes: Int, ...) → out self
Status: 1 change needed
Impact: Blocks confusion matrix tests
```

#### shared/core/types/fp8.mojo
```
Line 36: fn __init__(mut self, value: UInt8 = 0) → out self
Line 87: var mantissa = int(abs_x * 128.0) → Int(abs_x * 128.0)
Status: 2 changes needed
Impact: Blocks FP8 type usage in data tests
```

#### shared/core/types/integer.mojo
```
Line 39: fn __init__(mut self, value: Int8) → out self
Status: 1 change needed
Impact: Blocks Int8 type usage
```

#### shared/core/types/mxfp4.mojo
```
Line 57:  fn __init__(mut self, exponent: UInt8 = 127) → out self
Line 103: biased_exp.cast[DType.uint8]() → UInt8(biased_exp)
Line 474: fn __init__(mut self) → out self
Status: 3 changes needed
Impact: Blocks MXFP4 type usage in data tests
```

#### shared/core/types/nvfp4.mojo
```
Line 65:  fn __init__(mut self, value: UInt8 = 0x38) → out self
Line 94:  var mantissa = int(scale * 512.0) → Int(scale * 512.0)
Line 525: fn __init__(mut self) → out self
Status: 3 changes needed
Impact: Blocks NVFP4 type usage in data tests
```

### HIGH PRIORITY (Block Specific Test Suites)

#### tests/integration/test_all_architectures.mojo
```
Line 49: Invalid List mutation pattern
Pattern:
  List[Int]()
    .append(batch_size)
    .append(3)
    ...

Should be:
  var shape = List[Int]()
  shape.append(batch_size)
  shape.append(3)
  ...

Status: 1 change needed
Impact: Blocks entire integration test suite
```

#### tests/test_data_integrity.mojo
```
Lines 30, 51, 82, 107, 129, 190, 204, 233, 254:
  assert condition, "message" → Invalid syntax in v0.25.7+
  Should use: from testing import assert_equal, etc.

Line 181: Float32 → Float16 implicit conversion
  Should be: Float16(float32_value)

Line 167: Float64 → Float32 implicit conversion
  Should be: Float32(float64_value)

Status: 15+ changes needed
Impact: Blocks data integrity test execution
```

### MEDIUM PRIORITY (Test Infrastructure)

#### tests/shared/utils/test_*.mojo (6 files)
```
All files lack main() function
Files:
  - test_config.mojo
  - test_logging.mojo
  - test_io.mojo
  - test_profiling.mojo
  - test_random.mojo
  - test_visualization.mojo

Solution: Add main() function to each, OR
          Implement test runner framework, OR
          Update CI/CD workflow

Status: 6 files need modification
Impact: Blocks utility test execution
```

---

## Fix Complexity Analysis

### Quick Wins (10 minutes)

```
Change Type              | Count | Difficulty
========================|=======|============
Constructor param: mut→out | 10  | Trivial (regex replace)
int()→Int() conversion  |  2    | Trivial (regex replace)
List mutation pattern   |  1    | Simple (restructure code)
Type casting fix        |  1    | Simple (update method call)
```

**Total changes**: 14 (mostly mechanical)
**Estimated time**: 10-15 minutes

### Medium Tasks (20 minutes)

```
Fix Type                 | Count | Difficulty
========================|=======|============
Float type conversions  |  2    | Medium (logic understanding)
Assertion syntax update | 12+   | Medium (framework understanding)
```

**Total changes**: 14+
**Estimated time**: 20-30 minutes

### Infrastructure Tasks (30 minutes)

```
Task                           | Effort | Complexity
===============================|========|===========
Add main() to 6 test files    | Low    | Trivial
Implement test runner         | High   | Complex
Update CI/CD workflow         | Medium | Medium
```

---

## Execution Blockers Priority Queue

### Must Fix First (Blocks Everything)

1. **ExTensor constructor (2 changes)**
   ```
   shared/core/extensor.mojo:89,149
   fn __init__(mut self → out self
   ```
   Once fixed: 50% of tests can compile

2. **Metric constructors (5 changes)**
   ```
   shared/training/metrics/*.mojo
   All fn __init__(mut self → out self
   ```
   Once fixed: 80% of tests can compile

3. **Numeric type constructors (7 changes)**
   ```
   shared/core/types/*.mojo
   All fn __init__(mut self → out self
   ```
   Once fixed: 90% of tests can compile

### Then Fix (Unblock Specific Tests)

4. **List mutation in integration test (1 change)**
   ```
   tests/integration/test_all_architectures.mojo:49
   Restructure List building
   ```

5. **Deprecated int() calls (2 changes)**
   ```
   shared/core/types/fp8.mojo:87
   shared/core/types/nvfp4.mojo:94
   int() → Int()
   ```

6. **Type casting API (1 change)**
   ```
   shared/core/types/mxfp4.mojo:103
   .cast[DType.uint8]() → UInt8()
   ```

### Finally Address (Infrastructure)

7. **Test framework issues (6+ files)**
   ```
   Add main() or test runner
   6 utility test files
   ```

8. **Type conversion issues (2 changes)**
   ```
   tests/test_data_integrity.mojo:181,167
   Explicit Float conversions
   ```

9. **Assertion syntax (12+ instances)**
   ```
   tests/test_data_integrity.mojo
   Update to v0.25.7+ syntax
   ```

---

## Error Distribution by Severity

### Critical Blockers (Must Fix to Proceed)

```
Priority | Error Type                | Count | Status
---------|-------------------------|-------|--------
1        | Constructor param (mut) | 10    | NOT FIXED
2        | List mutation pattern   | 1     | NOT FIXED
3        | int() → Int() deprecated | 2     | NOT FIXED
4        | Type casting API change | 1     | NOT FIXED
```

**Total**: 14 critical errors across 11 files

### Important Issues (Prevent Test Execution)

```
Priority | Error Type              | Count | Status
---------|----------------------|-------|--------
5        | Type conversions (Float)| 2     | NOT FIXED
6        | Assertion syntax       | 12+   | NOT FIXED
7        | Test framework missing | 6     | NOT FIXED
```

**Total**: 20+ issues preventing test execution

---

## Recommended Fix Order

1. **Batch 1: Constructor Parameters (5 min)**
   - Find/replace: `fn __init__(mut self` → `fn __init__(out self`
   - Find/replace: `fn __copyinit__(mut self` → `fn __copyinit__(out self`
   - Files: 9 files, 14 changes
   - Expected outcome: 80% of compilation errors fixed

2. **Batch 2: Builtin Functions (2 min)**
   - Find/replace: `int(` → `Int(` (check context)
   - Files: 2 files, 2 changes
   - Expected outcome: All compilation errors fixed

3. **Batch 3: Type Casting (3 min)**
   - Fix `.cast[DType.uint8]()` → appropriate conversion
   - Files: 1 file, 1 change
   - Expected outcome: Data type compilation errors fixed

4. **Batch 4: List Pattern (5 min)**
   - Fix List chaining pattern in integration test
   - Files: 1 file, 1 change
   - Expected outcome: Integration test compiles

5. **Batch 5: Test Assertions (15 min)**
   - Update assertion syntax to v0.25.7+
   - Files: 1 file, 12+ changes
   - Expected outcome: Data integrity test compiles

6. **Batch 6: Type Conversions (5 min)**
   - Add explicit Float type conversions
   - Files: 2 files, 2 changes
   - Expected outcome: Data integrity test runs

7. **Batch 7: Test Framework (10 min)**
   - Add main() to utility tests or set up test runner
   - Files: 6+ files
   - Expected outcome: All utility tests can run

**Total estimated time**: 40-50 minutes for all compilation errors

