# Action Checklist - Example Validation

## Overview

57 Mojo example files analyzed. All fail compilation (0% success rate).
Root causes identified: Mojo language syntax changes, deprecated decorators, missing framework functionality.

## Quick Stats

- **Total Files Tested**: 57
- **Compilation Success**: 0/57 (0%)
- **Unique Error Categories**: 10
- **Files Needing Framework Fixes**: 9
- **Files Needing Syntax Fixes**: 40+
- **Estimated Fix Time**: 14-22 hours

## Phase 1: Framework Fixes (Do First)

### [ ] Task 1.1: Update ExTensor Struct
- **File**: `/shared/core/types/extensor.mojo`
- **Work**:
  - [ ] Add `@fieldwise_init` decorator
  - [ ] Implement `Copyable` trait via `__copyinit__`
  - [ ] Implement `Movable` trait via `__moveinit__`
  - [ ] Add `copy()` instance method
  - [ ] Add `full()` static factory method
  - [ ] Fix move semantics in return statements (use `^`)
- **Blocked By**: None
- **Blocks**: All examples using ExTensor
- **Est. Time**: 2-3 hours
- **Verification**:
  ```bash
  pixi run mojo build shared/core/types/extensor.mojo
  ```

### [ ] Task 1.2: Update FP8 Type Definition
- **File**: `/shared/core/types/fp8.mojo`
- **Work**:
  - [ ] Replace `@value` with `@fieldwise_init`
  - [ ] Update `__init__` to return `Self` (not use `inout self`)
  - [ ] Add `__copyinit__` method
  - [ ] Add `__moveinit__` method
  - [ ] Fix string conversion: `str()` → `String()`
  - [ ] Fix int conversion: `int()` → `Int()`
- **Blocked By**: None
- **Blocks**: fp8_example.mojo
- **Est. Time**: 1-2 hours
- **Verification**:
  ```bash
  pixi run mojo build shared/core/types/fp8.mojo
  ```

### [ ] Task 1.3: Update BF8 Type Definition
- **File**: `/shared/core/types/bf8.mojo`
- **Work**: Same as Task 1.2 (copy pattern)
- **Blocked By**: Task 1.2 (learn from FP8 fixes)
- **Blocks**: bf8_example.mojo
- **Est. Time**: 1 hour
- **Verification**:
  ```bash
  pixi run mojo build shared/core/types/bf8.mojo
  ```

### [ ] Task 1.4: Update Integer Type Definition
- **File**: `/shared/core/types/integer.mojo`
- **Work**: Same pattern as FP8/BF8 fixes
  - [ ] Replace `@value` with `@fieldwise_init`
  - [ ] Update `__init__` signatures
  - [ ] Add `__copyinit__` and `__moveinit__`
  - [ ] Fix string/int conversions
- **Blocked By**: Task 1.2 (pattern established)
- **Blocks**: integer_example.mojo
- **Est. Time**: 1-2 hours

### [ ] Task 1.5: Update Unsigned Integer Type Definition
- **File**: `/shared/core/types/unsigned.mojo`
- **Work**: Same as Task 1.4
- **Blocked By**: Task 1.4
- **Est. Time**: 1 hour

### [ ] Task 1.6: Fix Module Exports
- **File**: `/shared/core/__init__.mojo`
- **Work**:
  - [ ] Remove `__all__` file-scope list (or check Mojo 0.24.1+ syntax)
  - [ ] Explicitly export `Tensor` (alias from ExTensor or actual type)
  - [ ] Explicitly export `Module` base class
  - [ ] Explicitly export `Linear` layer
  - [ ] Ensure `creation` functions are accessible
  - [ ] Add clear import statements for all exports
- **Blocked By**: Tasks 1.1-1.5 (verify implementations exist)
- **Blocks**: attention_layer.mojo, prelu_activation.mojo, others
- **Est. Time**: 1-2 hours
- **Verification**:
  ```bash
  # Test that imports work
  pixi run mojo run -I . -c "from shared.core import Tensor, Module, Linear"
  ```

### [ ] Task 1.7: Fix Autograd Module Exports
- **File**: `/shared/autograd/__init__.mojo`
- **Work**:
  - [ ] Remove `__all__` file-scope list (line 63)
  - [ ] Let import statements define what's exported
  - [ ] Verify all submodules load correctly
- **Blocked By**: None
- **Blocks**: All autograd examples
- **Est. Time**: 1 hour
- **Verification**:
  ```bash
  pixi run mojo build shared/autograd/__init__.mojo
  ```

### [ ] Task 1.8: Fix Autograd Functional Module
- **File**: `/shared/autograd/functional.mojo`
- **Work**:
  - [ ] Replace `let` with `var` (line 66)
  - [ ] Check for other deprecated keywords
  - [ ] Fix move semantics in return statements
- **Blocked By**: None
- **Blocks**: simple_example.mojo, linear_regression*.mojo
- **Est. Time**: 1 hour

### [ ] Task 1.9: Fix Mixed Precision Training Module
- **File**: `/shared/training/mixed_precision.mojo`
- **Work**:
  - [ ] Remove `inout` from parameter lists (lines 258, 45)
  - [ ] Update function signatures to match Mojo 0.24.1+ style
  - [ ] Fix move semantics in returns
- **Blocked By**: Task 1.1 (ExTensor fixes)
- **Blocks**: mixed_precision_training.mojo
- **Est. Time**: 1 hour

**Subtotal Phase 1**: ~10-15 hours

---

## Phase 2: Module Verification

### [ ] Task 2.1: Verify/Create creation Module
- **Work**:
  - [ ] Check if `/shared/core/creation.mojo` exists
  - [ ] If missing, create with: `zeros()`, `ones()`, `full()`, `arange()`, `eye()`, `linspace()`
  - [ ] Ensure functions return ExTensor with correct shapes
  - [ ] Export from `shared/core/__init__.mojo`
- **Blocked By**: Task 1.6 (module exports)
- **Blocks**: All examples using tensor creation functions
- **Est. Time**: 2-3 hours if creating from scratch, 30 min if fixing existing

### [ ] Task 2.2: Find DynamicVector Location
- **Work**:
  - [ ] Search codebase for DynamicVector definition
  - [ ] Document correct import path
  - [ ] Update import statements in 8 affected files
  - [ ] If missing, consider alternative (List[Int], etc.)
- **Blocked By**: None
- **Blocks**: simd_example.mojo, simd_optimization.mojo, memory_optimization.mojo, basic_usage.mojo, test_arithmetic.mojo, fp8_example.mojo, bf8_example.mojo, trait_based_layer.mojo
- **Est. Time**: 1-2 hours
- **Commands**:
  ```bash
  find . -name "*.mojo" -exec grep -l "struct DynamicVector" {} \;
  find . -name "*.mojo" -exec grep -l "class DynamicVector" {} \;
  ```

### [ ] Task 2.3: Find simdwidthof Replacement
- **Work**:
  - [ ] Check Mojo 0.24.1+ documentation for SIMD width detection
  - [ ] Find replacement function or alternative approach
  - [ ] Update imports in 3 affected files
- **Blocked By**: None
- **Blocks**: simd_example.mojo, simd_optimization.mojo, performance examples
- **Est. Time**: 1 hour
- **Reference**: https://docs.modular.com/mojo/manual/parametric-types#simd

### [ ] Task 2.4: Verify Module and Linear Classes
- **Work**:
  - [ ] Confirm Module trait/class exists in `/shared/core/`
  - [ ] Confirm Linear layer class exists
  - [ ] If missing, create proper base classes
  - [ ] Add to exports in `__init__.mojo`
- **Blocked By**: Task 1.6
- **Blocks**: attention_layer.mojo, prelu_activation.mojo
- **Est. Time**: 2-3 hours if creating, 1 hour if fixing exports

**Subtotal Phase 2**: ~5-8 hours

---

## Phase 3: Example Syntax Fixes

### [ ] Task 3.1: Remove inout self from All Methods
- **Pattern**: `fn method(inout self, ...)` → `fn method(self, ...)`
- **Files Affected** (automated fix):
  ```
  trait_example.mojo (9 occurrences)
  ownership_example.mojo (3 occurrences)
  simd_example.mojo (2 occurrences)
  simd_optimization.mojo (2 occurrences)
  memory_optimization.mojo (2 occurrences)
  focal_loss.mojo (1 occurrence)
  attention_layer.mojo (3 occurrences)
  mixed_precision_training.mojo (2 occurrences)
  trait_based_layer.mojo (7 occurrences)
  And others...
  ```
- **Blocked By**: None (independent)
- **Est. Time**: 2-3 hours (automated find-replace + verification)
- **Commands**:
  ```bash
  # Find all occurrences
  grep -r "fn.*inout self" examples/

  # Automated fix (test on copies first)
  find examples -name "*.mojo" -type f -exec sed -i 's/fn \([a-zA-Z_][a-zA-Z0-9_]*\)(inout self,/fn \1(self,/g' {} \;
  find examples -name "*.mojo" -type f -exec sed -i 's/fn \([a-zA-Z_][a-zA-Z0-9_]*\)(inout self)/fn \1(self)/g' {} \;
  ```

### [ ] Task 3.2: Update import paths in examples
- **Work**:
  - [ ] basic_usage.mojo: `from src.extensor` → `from shared.core`
  - [ ] test_arithmetic.mojo: `from src.extensor` → `from shared.core`
  - [ ] Update all DynamicVector imports (use result from Task 2.2)
  - [ ] Update all simdwidthof imports (use result from Task 2.3)
- **Blocked By**: Tasks 2.1-2.3 (need correct paths)
- **Est. Time**: 1-2 hours

### [ ] Task 3.3: Fix String Conversions
- **Pattern**: `str(value)` → `String(value)` or similar
- **Files Affected**:
  ```
  fp8_example.mojo (2 occurrences)
  bf8_example.mojo (2 occurrences)
  integer_example.mojo (7 occurrences)
  ```
- **Blocked By**: Tasks 1.2-1.4 (framework fixes must work first)
- **Est. Time**: 1 hour

**Subtotal Phase 3**: ~4-6 hours

---

## Validation & Testing

### [ ] Task 4.1: Compile All Framework Files
- **After Phase 1 Complete**:
  ```bash
  pixi run mojo build shared/core/types/extensor.mojo
  pixi run mojo build shared/core/types/fp8.mojo
  pixi run mojo build shared/core/types/bf8.mojo
  pixi run mojo build shared/core/types/integer.mojo
  pixi run mojo build shared/core/types/unsigned.mojo
  pixi run mojo build shared/core/__init__.mojo
  pixi run mojo build shared/autograd/__init__.mojo
  pixi run mojo build shared/autograd/functional.mojo
  pixi run mojo build shared/training/mixed_precision.mojo
  ```
- **Est. Time**: 30 min

### [ ] Task 4.2: Test Simple Examples First
- **After Phase 2 Complete**:
  ```bash
  pixi run mojo run -I . examples/mojo-patterns/trait_example.mojo
  pixi run mojo run -I . examples/mojo-patterns/ownership_example.mojo
  pixi run mojo run -I . examples/basic_usage.mojo
  pixi run mojo run -I . examples/test_arithmetic.mojo
  ```
- **Est. Time**: 30 min

### [ ] Task 4.3: Test All Examples
- **After Phase 3 Complete**:
  ```bash
  # Run all examples and capture output
  cd /home/mvillmow/ml-odyssey
  for f in examples/**/*.mojo; do
    echo "Testing: $f"
    pixi run mojo run -I . "$f" 2>&1 | head -10
    echo "---"
  done > /tmp/example-test-results.txt
  ```
- **Expected Result**: All 57 files should compile and run
- **Est. Time**: 1-2 hours

### [ ] Task 4.4: Add CI/CD Validation
- **Work**:
  - [ ] Create GitHub Actions workflow to validate examples
  - [ ] Run on every PR to catch regressions
  - [ ] Document in README how to run examples locally
- **Est. Time**: 1-2 hours

**Subtotal Phase 4**: ~3-4 hours

---

## Grand Total

| Phase | Task | Hours |
|-------|------|-------|
| 1 | Framework Fixes | 10-15 |
| 2 | Module Verification | 5-8 |
| 3 | Example Syntax Fixes | 4-6 |
| 4 | Validation & Testing | 3-4 |
| **TOTAL** | | **22-33 hours** |

## Critical Path (Minimum to Get One Example Working)

1. **2 hours**: Fix ExTensor struct (Task 1.1) - CRITICAL BLOCKER
2. **1 hour**: Fix module exports (Task 1.6) - Needed for imports
3. **1 hour**: Find DynamicVector location (Task 2.2) - Needed for collections
4. **1 hour**: Remove inout self from one example (Task 3.1) - Syntax fix
5. **30 min**: Test the example (Task 4.1) - Verification

**Critical Path Total**: ~5.5 hours to get first example compiling

## Recommended Approach

### Week 1: Foundation (Days 1-3)
- [ ] Complete Phase 1 (Tasks 1.1-1.9): Framework Fixes
- [ ] Complete Phase 2 (Tasks 2.1-2.4): Module Verification
- **Outcome**: Framework is modern and modules are in place

### Week 1-2: Examples (Days 4-5+)
- [ ] Complete Phase 3 (Tasks 3.1-3.3): Example Syntax
- [ ] Complete Phase 4 (Tasks 4.1-4.4): Validation
- **Outcome**: All examples compile and run

## Known Risks

1. **DynamicVector Location Unknown** (Task 2.2)
   - May need to implement custom Vector type
   - Could delay other fixes

2. **Module Base Class Missing** (Task 2.4)
   - May need to create from scratch
   - Need to design proper interface

3. **Mojo Version Compatibility**
   - Some fixes may differ between Mojo versions
   - Check `mojo --version` before starting

4. **Cascading Errors**
   - One framework fix may expose new errors
   - Plan for iterative fixing, not linear

## Success Criteria

- [ ] All 57 example files compile without errors
- [ ] Examples run and produce expected output
- [ ] No compiler warnings
- [ ] All imports resolve correctly
- [ ] All trait conformances properly implemented
- [ ] CI/CD validation passing
- [ ] Examples documented in README
- [ ] Future regressions prevented by CI

## Rollback Plan (If needed)

- Keep original files in git history
- Tag before/after versions
- Can revert individual fixes if they break something else
- Test on feature branch before merging to main

---

**Created**: 2025-11-22
**Status**: Ready for implementation
**Next Step**: Begin Phase 1, Task 1.1 (ExTensor struct update)
