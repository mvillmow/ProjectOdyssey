# FIXME Guide: Data Test Compilation Errors

Quick reference guide for fixing the 13 categories of compilation errors blocking data tests.

## Quick Fix Summary (Estimated 4-6 hours)

| Category | Files | Fixes | Est. Time |
|----------|-------|-------|-----------|
| @value decorator | 4 impl files | Replace with @fieldwise_init + traits | 30 min |
| ExTensor memory | 2 impl files | Add ^ for moves, add dtype | 45 min |
| num_elements() | 2 impl files | Add method or implement inline | 30 min |
| Trait conformances | 2 impl files | Add (Copyable, Movable) | 15 min |
| owned keyword | 4 functions | Replace with var | 10 min |
| String conversion | 3 tests | Implement str() helper | 20 min |
| Type mismatches | 5 locations | Fix type casting | 20 min |
| Struct inheritance | 1 file | Use composition | 15 min |
| Test main() | 3 files | Add orchestrator functions | 20 min |
| Parameter ordering | 2 functions | Reorder parameters | 10 min |
| StringSlice | 1 function | Explicit conversion | 5 min |
| Dynamic traits | 1 file | Use generic pattern | 20 min |
| Type inference | 1 function | Fix type casting | 10 min |

---

## Category 1: Fix @value Decorator

### Before (Current - Won't Compile)

```mojo
@value
struct Normalize:
    var mean: Float32
    var std: Float32
```

### After (Fixed)

```mojo
@fieldwise_init
struct Normalize(Copyable, Movable):
    var mean: Float32
    var std: Float32
```

### Files to Update (14 structs total)

**transforms.mojo** (4 structs):

- Line 405: `Normalize`
- Line 501: `Denormalize`
- Line 629: `RandomCrop`
- Line 728: `RandomRotation`

**text_transforms.mojo** (4 structs):

- Line 113: `RandomSwapWords`
- Line 178: `RandomDeleteWords`
- Line 242: `RandomDuplicateWords`
- Line 319: `SynonymReplacement`

**generic_transforms.mojo** (6 structs):

- Line 38: `IdentityTransform`
- Line 70: `Compose`
- Line 178: `Clamp`
- Line 242: `Normalize`
- Line 411: `ToFloat32`
- Line 445: `ToInt32`

**samplers.mojo** (3 structs):

- Line 38: `SequentialSampler`
- Line 91: `RandomSampler`
- Line 172: `WeightedSampler`

---

## Category 2: Fix ExTensor Memory Management

### Pattern 2a: Add Move Semantics in Returns

**Before**:

```mojo
fn crop(self, h_start: Int, h_end: Int, w_start: Int, w_end: Int) -> ExTensor:
    var cropped = ExTensor(...)
    # ... cropping logic ...
    return cropped  # ERROR: cannot be implicitly copied
```

**After**:

```mojo
fn crop(self, h_start: Int, h_end: Int, w_start: Int, w_end: Int) -> ExTensor:
    var cropped = ExTensor(...)
    # ... cropping logic ...
    return cropped^  # Move ownership
```

**Files & Lines** (transforms.mojo):

- Line 539: `erase()` method
- Line 603: `rotate_90()` method
- Line 788: `to_grayscale()` method
- Line 825: `to_rgb()` method
- Line 833: `invert()` method

### Pattern 2b: Add dtype Parameter in ExTensor Constructor

**Before**:

```mojo
return ExTensor(cropped^)  # ERROR: missing dtype
```

**After**:

```mojo
return ExTensor(cropped^, DType.float32)
```

**Files & Lines** (transforms.mojo):

- Line 498: `Crop.apply()` method
- Line 402: Similar location in crop transform

**Alternative Fix**: Update ExTensor.**init** signature:

```mojo
fn __init__(out self, data: List[Float32], shape: List[Int], dtype: DType = DType.float32) raises:
    # Allow dtype to be optional with default
```

### Pattern 2c: Add num_elements() Method to ExTensor

**Option 1: Add to ExTensor (preferred)**

File: `/home/mvillmow/ml-odyssey/shared/core/extensor.mojo`

```mojo
struct ExTensor:
    # ... existing fields ...

    fn num_elements(self) -> Int:
        """Return total number of elements in tensor."""
        var count = 1
        for dim in self.shape:
            count *= dim
        return count
```

**Option 2: Fix all callers** (if Option 1 not possible)

Replace each `data.num_elements()` with:

```mojo
var num_elements = 1
for dim in data.shape:
    num_elements *= dim
```

**Files & Lines** (transforms.mojo):

- Line 51: `random_float()` helper
- Line 681, 546, 795, 610: Various transform methods
- Line 833: `normalize()` method

**Files & Lines** (generic_transforms.mojo):

- Lines 108, 110, 221, 223, 278, 281, 437, 439, 473, 475

---

## Category 3: Fix Trait Conformances

### Before

```mojo
trait Transform:
    fn apply(self, data: ExTensor) -> ExTensor:
        ...

# This fails - Transform not Copyable/Movable
struct Pipeline:
    var transforms: List[Transform]  # ERROR
```

### After

```mojo
trait Transform(Copyable, Movable):
    fn apply(self, data: ExTensor) -> ExTensor:
        ...

# Now this works - Transform is Copyable/Movable
struct Pipeline:
    var transforms: List[Transform]  # OK
```

**Files to Update** (2 files, 2 traits):

1. `/home/mvillmow/ml-odyssey/shared/data/transforms.mojo:54`
   - Trait: `Transform`
   - Fix: Change `trait Transform:` to `trait Transform(Copyable, Movable):`

2. `/home/mvillmow/ml-odyssey/shared/data/text_transforms.mojo:65`
   - Trait: `TextTransform`
   - Fix: Change `trait TextTransform:` to `trait TextTransform(Copyable, Movable):`

---

## Category 4: Remove Deprecated `owned` Keyword

### Before

```mojo
fn __moveinit__(out self, owned existing: Self):
    # ERROR: 'owned' has been deprecated, use 'deinit' instead
```

### After

```mojo
fn __moveinit__(out self, deinit existing: Self):
    # For move initialization
```

**For function parameters**, use `var` instead:

### Before

```mojo
fn __init__(out self, p: Float64 = 0.1, owned vocabulary: List[String]):
    # ERROR: 'owned' is deprecated and also has wrong parameter ordering
```

### After

```mojo
fn __init__(out self, vocabulary: List[String], p: Float64 = 0.1):
    self.vocabulary = vocabulary^  # Transfer ownership
```

**Files & Lines to Update**:

1. `/home/mvillmow/ml-odyssey/shared/data/loaders.mojo:40`
   - Change: `owned existing` → `deinit existing`

2. `/home/mvillmow/ml-odyssey/shared/data/text_transforms.mojo:257, 333`
   - Change: `owned vocabulary/synonyms` → remove param, reorder (see Category 10)

3. `/home/mvillmow/ml-odyssey/shared/data/samplers.mojo:186`
   - Change: `owned weights` → `var weights`

---

## Category 5: Implement String Conversion

### Add Helper Function (in test files)

File: Add to each test file that uses `str()`:

```mojo
fn int_to_string(value: Int) -> String:
    """Convert integer to string.

    Simple implementation - works for common cases.
    """
    if value == 0:
        return "0"

    var negative = value < 0
    var abs_val = abs(value)
    var result = ""

    while abs_val > 0:
        var digit = abs_val % 10
        result = String(digit) + result
        abs_val = abs_val // 10

    if negative:
        result = "-" + result

    return result
```

**Usage**:

```mojo
# Before
file_paths.append("/path/to/image_" + str(i) + ".jpg")

# After
file_paths.append("/path/to/image_" + int_to_string(i) + ".jpg")
```

**Files to Update**:

- `test_file_dataset.mojo`: Lines 104, 169
- `test_random.mojo`: Line 173

---

## Category 6: Fix Type Mismatches

### Location 1: Sampler Type Inference (samplers.mojo:251)

**Before**:

```mojo
var r = random_si64(0, Int64(total)) as Float64  # r is Float64
var cumsum = List[Int64](...)  # cumsum contains Int64

if r < cumsum[i]:  # ERROR: can't compare Float64 < Int64
```

**After**:

```mojo
var r = Float64(random_si64(0, Int64(total))) / Float64(total)
var cumsum = List[Float64](...)  # Convert to Float64

if r < cumsum[i]:  # OK: Float64 < Float64
```

---

## Category 7: Fix Struct Inheritance

### Before (Mojo doesn't support struct inheritance)

```mojo
struct BaseLoader:
    var dataset: Dataset
    var batch_size: Int

struct BatchLoader(BaseLoader):  # ERROR: inheriting from structs not allowed
    var shuffle: Bool
```

### After (Use Composition)

```mojo
struct BaseLoader:
    var dataset: Dataset
    var batch_size: Int

struct BatchLoader:
    var base: BaseLoader  # Composition instead of inheritance
    var shuffle: Bool

    fn get_dataset(self) -> Dataset:
        return self.base.dataset
```

**File**: `/home/mvillmow/ml-odyssey/shared/data/loaders.mojo:106`

---

## Category 8: Fix Parameter Ordering

### Before

```mojo
fn __init__(out self, p: Float64 = 0.1, n: Int = 1, owned vocabulary: List[String]):
    # ERROR: required parameter 'vocabulary' follows optional parameters
```

### After

```mojo
fn __init__(out self, vocabulary: List[String], p: Float64 = 0.1, n: Int = 1):
    self.vocabulary = vocabulary^
```

**Files & Lines**:

- `/home/mvillmow/ml-odyssey/shared/data/text_transforms.mojo:257`
  - Function: `RandomSwapWords.__init__`
  - Params: Move `vocabulary` before optional `p`, `n`

- `/home/mvillmow/ml-odyssey/shared/data/text_transforms.mojo:333`
  - Function: `SynonymReplacement.__init__`
  - Params: Move `synonyms` before optional `p`

---

## Category 9: Fix StringSlice Conversion

### Before

```mojo
var parts = text.split(" ")
for i in range(parts.size()):
    words.append(parts[i])  # ERROR: StringSlice can't implicitly convert to String
```

### After

```mojo
var parts = text.split(" ")
for i in range(parts.size()):
    words.append(String(parts[i]))  # Explicit conversion to String
```

**File**: `/home/mvillmow/ml-odyssey/shared/data/text_transforms.mojo:84`

---

## Category 10: Fix String Slice Copying

### Before

```mojo
var words = new_words  # ERROR: can't implicitly copy List[String]
```

### After

```mojo
words = new_words^  # Transfer ownership
```

**File**: `/home/mvillmow/ml-odyssey/shared/data/text_transforms.mojo:314, 372`

---

## Category 11: Fix Dynamic Trait Usage

### Before (Won't compile)

```mojo
struct BaseLoader:
    var dataset: Dataset  # ERROR: dynamic traits not supported
```

### After (Use Generic Type Parameter)

```mojo
struct BaseLoader[DatasetType: Dataset]:
    var dataset: DatasetType

    fn get_dataset(self) -> DatasetType:
        return self.dataset
```

**File**: `/home/mvillmow/ml-odyssey/shared/data/loaders.mojo:60`

---

## Category 12: Fix Optional Value Access

### Before

```mojo
var pad = self.padding.value()[]  # ERROR: Int is not subscriptable
```

### After

```mojo
var pad = self.padding.value()  # value() returns the Int directly
```

**Files & Lines** (transforms.mojo):

- Line 452: `Pad.apply()` method
- Line 473: Similar location

---

## Category 13: Add Test Main Functions

### Before (Won't compile - no main function)

```mojo
# test_datasets.mojo
fn test_tensor_dataset():
    ...

fn test_file_dataset():
    ...
```

### After (Add main orchestrator)

```mojo
fn test_tensor_dataset():
    ...

fn test_file_dataset():
    ...

fn main():
    print("Running dataset tests...")
    test_tensor_dataset()
    test_file_dataset()
    print("All dataset tests passed!")
```

**Files to Update**:

- `/home/mvillmow/ml-odyssey/tests/shared/data/test_datasets.mojo`
- `/home/mvillmow/ml-odyssey/tests/shared/data/test_loaders.mojo`
- `/home/mvillmow/ml-odyssey/tests/shared/data/test_transforms.mojo`

---

## Implementation Order

### Phase 1: Unblock Compilation (1-2 hours)

1. Replace all @value with @fieldwise_init + trait conformances (30 min)
2. Add (Copyable, Movable) to Transform and TextTransform traits (15 min)
3. Remove all `owned` keywords (10 min)
4. Fix parameter ordering (10 min)
5. Add ^ move semantics to ExTensor returns (20 min)

### Phase 2: Fix Type System (1-2 hours)

6. Add dtype to ExTensor constructors (30 min)
2. Implement num_elements() method (30 min)
3. Fix struct inheritance with composition (15 min)
4. Fix type mismatches in samplers (20 min)

### Phase 3: Fix Utilities (30-45 min)

10. Implement string conversion functions (20 min)
2. Fix StringSlice conversions (10 min)
3. Fix optional value access (5 min)

### Phase 4: Test Infrastructure (30-45 min)

13. Add main() functions to wrapper tests (20 min)
2. Run all tests to verify fixes (15 min)

---

## Verification Checklist

After implementing fixes:

- [ ] test_base_dataset.mojo still passes
- [ ] test_base_loader.mojo passes (no warnings)
- [ ] test_parallel_loader.mojo still passes
- [ ] test_pipeline.mojo still passes
- [ ] test_tensor_transforms.mojo still passes
- [ ] test_image_transforms.mojo still passes
- [ ] test_tensor_dataset.mojo passes
- [ ] test_file_dataset.mojo passes
- [ ] test_batch_loader.mojo passes
- [ ] test_augmentations.mojo passes
- [ ] test_text_augmentations.mojo passes
- [ ] test_generic_transforms.mojo passes
- [ ] test_sequential.mojo passes
- [ ] test_random.mojo passes
- [ ] test_weighted.mojo passes
- [ ] test_datasets.mojo passes
- [ ] test_loaders.mojo passes
- [ ] test_transforms.mojo passes
- [ ] run_all_tests.mojo passes with all 38 tests

---

## Reference: @value to @fieldwise_init Migration

Common pattern for the 14 structs:

```mojo
# Pattern for simple value structs
@fieldwise_init
struct MyStruct(Copyable, Movable):
    var field1: Int
    var field2: String
    var field3: Float32

    # Explicit initialization if needed
    fn __init__(out self, field1: Int, field2: String, field3: Float32):
        self.field1 = field1
        self.field2 = field2
        self.field3 = field3

# Pattern for structs with methods
@fieldwise_init
struct TransformStruct(Copyable, Movable):
    var param: Float32

    fn apply(self, data: ExTensor) -> ExTensor:
        # Implementation
        ...
```

---

**Guide Last Updated**: 2025-11-22
**Estimated Total Time**: 4-6 hours for all fixes
**Priority**: P1 - Blocks all comprehensive testing
