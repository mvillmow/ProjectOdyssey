# Agent Mojo Guidelines - v0.25.7+ Update

**Date**: 2025-01-23
**Purpose**: Update all agent configurations to prevent Mojo v0.25.7+ syntax errors
**Based On**: [mojo-v0.25.7-migration-errors.md](./mojo-v0.25.7-migration-errors.md)

## Critical Guidelines (MUST ALWAYS FOLLOW)

### 1. Constructor Signatures

✅ **ALWAYS use `out self` in `__init__` methods**

```mojo
// ❌ WRONG:
fn __init__(mut self):
fn __init__(mut self, value: Int):

// ✅ CORRECT:
fn __init__(out self):
fn __init__(out self, value: Int):
```

**Why**: Mojo requires constructors to return `Self` type with `out` argument.

**Agent Action**: Search codebase for `fn __init__(mut self` and replace with `out self`.

---

### 2. Parameter Conventions

✅ **Use `mut` for mutable parameters, NEVER `inout`**

```mojo
// ❌ WRONG:
fn modify(inout data: ExTensor):
fn process(inout self):

// ✅ CORRECT:
fn modify(mut data: ExTensor):
fn process(mut self):
```

**Why**: `inout` keyword was renamed to `mut` in v0.25.7+.

**Agent Action**: Replace all `inout` with `mut`.

---

### 3. String Conversion

✅ **Use `String()` constructor, NEVER `str()` function**

```mojo
// ❌ WRONG:
var s = str(value)
var msg = "Error: " + str(code)

// ✅ CORRECT:
var s = String(value)
var msg = "Error: " + String(code)
```

**Why**: `str()` function was deprecated and removed.

**Agent Action**: Replace all `str(` with `String(`.

---

### 4. Struct Traits

✅ **ALWAYS add explicit traits to structs**

```mojo
// ❌ WRONG:
struct MetricResult:
    var value: Float64

// ✅ CORRECT:
struct MetricResult(Copyable, Movable):
    var value: Float64
```

**Why**: Structs used in List, parameters, or return values must explicitly implement Copyable & Movable.

**When Required**:
- Struct stored in `List[StructType]`
- Struct passed as function parameter
- Struct returned from function
- Struct used in collections

**Agent Action**: Add `(Copyable, Movable)` to ALL struct definitions unless there's a specific reason not to.

---

### 5. Decorator Usage

✅ **Use `@fieldwise_init` with traits, NOT `@value`**

```mojo
// ❌ WRONG:
@value
struct Point:
    var x: Float32
    var y: Float32

// ✅ CORRECT:
@fieldwise_init
struct Point(Copyable, Movable):
    var x: Float32
    var y: Float32
```

**Why**: `@value` decorator was removed.

**Agent Action**: Replace `@value` with `@fieldwise_init` + explicit traits.

---

✅ **NEVER use both `@fieldwise_init` and manual `__init__`**

```mojo
// ❌ WRONG:
@fieldwise_init
struct Dataset(Copyable, Movable):
    var size: Int

    fn __init__(out self, size: Int):
        self.size = size

// ✅ CORRECT (choose one):

// Option 1: Auto-generated init
@fieldwise_init
struct Dataset(Copyable, Movable):
    var size: Int

// Option 2: Manual init
struct Dataset(Copyable, Movable):
    var size: Int

    fn __init__(out self, size: Int):
        self.size = size
```

**Why**: Having both causes conflicts - decorator auto-generates an init.

**Agent Action**: Remove `@fieldwise_init` if manual `__init__` exists.

---

### 6. Ownership & Copying

✅ **Use ownership transfer `^` for ExTensor and List assignments**

```mojo
// ❌ WRONG (implicit copy not allowed):
var copy = some_tensor
self.field = parameter
return result

// ✅ CORRECT (ownership transfer):
var copy = some_tensor^
self.field = parameter^
return result^
```

**Why**: ExTensor, List[T] are no longer ImplicitlyCopyable.

**When to use `^`**: When value is not needed after assignment (most cases).

**Agent Action**: Add `^` to all ExTensor/List assignments unless explicit copy is intended.

---

✅ **Use explicit copy constructors when copy is needed**

```mojo
// ❌ WRONG:
var copy = some_list  # Tries implicit copy

// ✅ CORRECT:
var copy = List[Int](some_list)  # Explicit copy
var copy = ExTensor(original)  # Explicit copy
```

**When to use**: When original value must remain valid after "copy".

**Agent Action**: Use copy constructor when value is used after assignment.

---

### 7. Struct Composition (NOT Inheritance)

✅ **Use composition, NEVER struct inheritance**

```mojo
// ❌ WRONG (not supported):
struct BatchLoader(BaseLoader, Copyable, Movable):
    pass

// ✅ CORRECT (composition):
struct BatchLoader(Copyable, Movable):
    var dataset: Dataset
    var batch_size: Int
    var drop_last: Bool
    # Copy all needed fields from BaseLoader

    fn __init__(out self, dataset: Dataset, batch_size: Int):
        self.dataset = dataset^
        self.batch_size = batch_size
        self.drop_last = False
```

**Why**: Mojo does not support struct inheritance.

**Agent Action**: Replace inheritance with composition - copy fields to child struct.

---

### 8. Compile-Time Generics for Traits

✅ **Use parametric generics `[T: Trait]`, NOT trait-typed fields**

```mojo
// ❌ WRONG (dynamic traits not supported):
struct Container:
    var transform: Transform  # Runtime trait field

// ✅ CORRECT (compile-time generic):
struct Container[T: Transform]:
    var transform: T
```

**Why**: Mojo doesn't support runtime trait dispatch - requires compile-time generics.

**Agent Action**: Convert trait fields to generic parameters.

---

## High Priority Guidelines (Common Patterns)

### 9. Import Statements

✅ **Import from correct modules**

```mojo
// ❌ WRONG:
from sys import DType
from sys import simdwidthof
from memory import UnsafePointer  # Missing Pointer

// ✅ CORRECT:
from memory import DType
from sys.info import simdwidthof
from memory import UnsafePointer, Pointer
```

**Module Changes**:
- `DType`: `sys` → `memory`
- `simdwidthof`: `sys` → `sys.info`
- `Pointer`: Import alongside `UnsafePointer`

**Agent Action**: Update import statements per module reorganization.

---

✅ **Don't import builtins**

```mojo
// ❌ WRONG (now builtins):
from collections import Tuple
from math import abs, round, max, min, pow

// ✅ CORRECT (remove imports):
# Tuple, abs, round, max, min are builtins - no import needed
# Use ** operator instead of pow
```

**Builtins** (no import needed):
- `Tuple`
- `abs`, `round`
- `max`, `min`

**Agent Action**: Remove these from import statements.

---

### 10. ExTensor API Changes

✅ **Use standalone functions, NOT instance methods**

```mojo
// ❌ WRONG (methods removed):
var result = ExTensor.from_scalar(value, dtype)
var total = tensor.sum()
var result = a.matmul(b)

// ✅ CORRECT (standalone functions):
from shared.core.extensor import full
from shared.core.reduction import sum as tensor_sum
from shared.core.matrix import matmul

var result = full(shape, value, dtype)
var total = tensor_sum(tensor)
var result = matmul(a, b)
```

**Removed Methods**:
- `ExTensor.from_scalar()` → `full()`
- `tensor.sum()` → `tensor_sum()`
- `tensor.matmul()` → `matmul()`

**Agent Action**: Use function-based API, add necessary imports.

---

✅ **ExTensor() constructor requires arguments**

```mojo
// ❌ WRONG:
self.tensor = ExTensor()

// ✅ CORRECT:
self.tensor = ExTensor(List[Float32](), DType.float32)
```

**Agent Action**: Always provide data and dtype to ExTensor constructor.

---

### 11. Initialization Functions

✅ **Always provide explicit dtype parameter**

```mojo
// ❌ WRONG (missing dtype):
var tensor = zeros(shape)
var tensor = ones(shape)
var tensor = full(shape, value)

// ✅ CORRECT:
var tensor = zeros(shape, DType.float32)
var tensor = ones(shape, DType.float32)
var tensor = full(shape, value, DType.float32)
```

**Affected Functions**: `zeros()`, `ones()`, `full()`, `empty()`

**Agent Action**: Add explicit dtype to all initialization function calls.

---

### 12. Float64 Constants

✅ **Use arithmetic expressions, NOT class methods**

```mojo
// ❌ WRONG (removed):
var nan = Float64.nan
var inf = Float64.inf
var neg_inf = Float64.neg_inf

// ✅ CORRECT:
var nan = Float64(0.0) / Float64(0.0)  # NaN
var inf = Float64(1.0) / Float64(0.0)  # +Inf
var neg_inf = -Float64(1.0) / Float64(0.0)  # -Inf
```

**Agent Action**: Replace Float64.* constants with arithmetic expressions.

---

### 13. Pointer API

✅ **Use Pointer.address_of(), NOT UnsafePointer.address_of()**

```mojo
// ❌ WRONG:
var ptr = UnsafePointer.address_of(variable)
var value = ptr.bitcast[Type]()[0]

// ✅ CORRECT:
from memory import Pointer
var ptr = Pointer.address_of(variable)
var value = ptr.bitcast[Type]()[]  # Note: [] not [0]
```

**Changes**:
1. Use `Pointer` type, not `UnsafePointer` for address_of
2. Array access: `[]` not `[0]`

**Agent Action**: Update pointer usage and import Pointer.

---

## Medium Priority Guidelines (Context-Dependent)

### 14. Shape: Property vs Method

⚠️ **Check whether shape is property or method in context**

**In most files** (after automated fix):
```mojo
var s = tensor.shape  # Property access
var dim = tensor.shape[0]  # Index property
```

**In matrix.mojo and extensor.mojo**:
```mojo
var s = tensor.shape()  # Method call
var dim = tensor.shape()[0]  # Index method result
```

**Why**: Inconsistent - extensor.mojo defines it as method, but many files use it as property.

**Agent Action**:
1. Check file context
2. If in doubt, try `.shape()` first (method call)
3. If error "expects 0 parameters", use `.shape` (property)

---

### 15. SIMD Type Inference

⚠️ **Keep SIMD types consistent in operations**

```mojo
// ❌ WRONG (mixed types):
return Scalar[T](1.0) / (Scalar[T](1.0) + exp(-Float32(x)))
# Mixes Scalar[T] with Float32

// ✅ CORRECT (consistent types):
@parameter
if T == DType.float16:
    # Do all computation in Float32, then cast
    var x_f32 = Float32(x)
    var result_f32 = Float32(1.0) / (Float32(1.0) + exp(-x_f32))
    return Scalar[T](result_f32)
else:
    # Use generic type throughout
    return Scalar[T](1.0) / (Scalar[T](1.0) + exp(-x))
```

**Why**: Mixing generic and explicit types causes inference failures.

**Agent Action**: Use parametric `@parameter if` branches for type-specific paths.

---

### 16. len() Type Resolution

⚠️ **Assign method results to variable before calling len()**

```mojo
// ❌ WRONG (type inference fails):
if len(tensor.shape()) == 2:

// ✅ CORRECT:
var shape_vec = tensor.shape()
if len(shape_vec) == 2:
```

**Why**: len() needs explicit type info when called on method results.

**Agent Action**: Assign to variable first, then call len().

---

### 17. DType Comparison

⚠️ **Use manual comparison, NOT assert_equal**

```mojo
// ❌ WRONG:
assert_equal(tensor.dtype, DType.float32, "message")

// ✅ CORRECT:
if tensor.dtype() != DType.float32:
    raise Error("message")
```

**Why**: DType doesn't implement Comparable trait.

**Agent Action**: Manual comparison in tests.

---

### 18. raises Declaration

⚠️ **Add `raises` to functions calling raising code**

```mojo
// ❌ WRONG:
fn __init__(out self, size: Int):
    self.data = ExTensor(shape, DType.int32)  # Can raise

// ✅ CORRECT:
fn __init__(out self, size: Int) raises:
    self.data = ExTensor(shape, DType.int32)
```

**Why**: Functions calling raising code must declare `raises`.

**Agent Action**: Add `raises` if calling any function that raises.

---

## Agent-Specific Instructions

### For `implementation-engineer` Agents

When writing new Mojo code:

1. ✅ ALL structs get `(Copyable, Movable)` traits by default
2. ✅ ALL `__init__` methods use `out self`
3. ✅ Use `@fieldwise_init` for simple structs without validation
4. ✅ Use `String()` not `str()`, `mut` not `inout`
5. ✅ Transfer ownership with `^` for ExTensor/List assignments
6. ✅ Import from correct modules (DType from memory, etc.)
7. ✅ Use standalone functions for ExTensor operations
8. ✅ Always provide dtype to initialization functions
9. ✅ No struct inheritance - use composition
10. ✅ Use compile-time generics for traits

### For `implementation-specialist` Agents

When planning implementations:

1. ✅ Specify ownership patterns in design (transfer vs copy)
2. ✅ Document which structs need traits
3. ✅ Plan composition instead of inheritance
4. ✅ Identify generic parameters for trait constraints
5. ✅ Note which functions need `raises` declaration

### For `mojo-language-review-specialist` Agents

When reviewing code:

1. ✅ Check ALL `__init__` use `out self`
2. ✅ Verify structs have `(Copyable, Movable)` traits
3. ✅ Ensure no `@value` decorator usage
4. ✅ Check for `^` ownership transfer on ExTensor/List
5. ✅ Verify correct imports (DType, simdwidthof, Pointer)
6. ✅ Check no struct inheritance
7. ✅ Verify no trait fields (use generics instead)
8. ✅ Check SIMD type consistency
9. ✅ Verify explicit dtype in initialization calls

### For `code-review-orchestrator` Agents

Add these checks to review workflow:

1. ✅ Run search for `fn __init__(mut self` → flag for correction
2. ✅ Run search for `@value` → flag for @fieldwise_init conversion
3. ✅ Run search for `str(` → flag for String() conversion
4. ✅ Check struct definitions have traits
5. ✅ Verify no implicit ExTensor/List copies

---

## Pre-Commit Check Suggestions

Add these checks to `.pre-commit-config.yaml`:

```yaml
- id: mojo-syntax-check
  name: Mojo v0.25.7+ Syntax Check
  entry: python scripts/check_mojo_syntax.py
  language: python
  files: \.mojo$
```

**Check script should flag**:
1. `fn __init__(mut self`
2. `@value`
3. `str(`
4. `from sys import DType`
5. `from collections import Tuple`
6. `struct.*:` without `(Copyable, Movable)`
7. ExTensor/List assignment without `^` or explicit copy

---

## Migration Checklist for Future Mojo Updates

When Mojo updates again:

1. ✅ Document ALL compilation errors systematically
2. ✅ Categorize errors by type (syntax, API, ownership, etc.)
3. ✅ Create comprehensive error catalog (like this document)
4. ✅ Update agent configurations BEFORE bulk code generation
5. ✅ Add pre-commit checks for common errors
6. ✅ Test on small subset before applying to full codebase
7. ✅ Use automation (Python scripts) for repetitive fixes
8. ✅ Commit fixes in logical batches for easier review

---

## Summary: Top 10 Must-Follow Rules

1. ✅ **`out self`** in `__init__`
2. ✅ **`String()` not `str()`**
3. ✅ **`mut` not `inout`**
4. ✅ **`(Copyable, Movable)` on all structs**
5. ✅ **`@fieldwise_init` not `@value`**
6. ✅ **Ownership transfer `^` for ExTensor/List**
7. ✅ **Composition not inheritance**
8. ✅ **Generics `[T: Trait]` not trait fields**
9. ✅ **Explicit dtype in init functions**
10. ✅ **Correct imports** (DType from memory, etc.)

## Files to Update

**Agent Configuration Files**: Update all agent YAML frontmatter with these guidelines:
- `.claude/agents/**/*.md`
- Especially: implementation-engineer, implementation-specialist, mojo-language-review-specialist

**Add to CLAUDE.md**: Link to this document in Mojo syntax section.

**Skills to Update**:
- `mojo-format` skill
- `mojo-memory-check` skill
- Any code generation skills

---

**Next Actions**:
1. Update `.clinerules` with link to this document
2. Update `CLAUDE.md` Mojo syntax section
3. Update agent configurations
4. Create pre-commit hook script
5. Test on small code sample
