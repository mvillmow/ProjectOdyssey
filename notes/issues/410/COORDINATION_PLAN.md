# Image Augmentations Implementation Coordination Plan

## Executive Summary

This document coordinates the implementation of image augmentation transforms across issues #409 (Tests) and #410 (Implementation). The work involves resolving 7 TODOs, implementing 2 missing transforms, and enabling 14 currently-commented test functions.

## Current State

### Test File Status
**File**: `/home/user/ml-odyssey/tests/shared/data/transforms/test_augmentations.mojo`
- 14 comprehensive test functions defined
- ALL tests currently commented out (placeholder `pass` statements)
- Tests ready to verify implementation once features are complete

### Implementation File Status
**File**: `/home/user/ml-odyssey/shared/data/transforms.mojo`
- Partial implementations with 7 TODOs
- Missing transforms: RandomErasing, RandomVerticalFlip
- Missing type: Pipeline (tests expect this name)
- Existing transforms work on 1D data, need 2D image support

## Gap Analysis

### Critical Gaps

| Gap | Test Expectations | Current Implementation | Impact |
|-----|-------------------|------------------------|---------|
| **RandomErasing** | Full implementation with scale parameter | Missing entirely | 2 tests fail |
| **RandomVerticalFlip** | Full implementation | Missing entirely | Design gap |
| **Pipeline type** | Tests use `Pipeline([...])` | Only `Compose` exists | 2 tests fail |
| **RandomRotation** | Actual rotation | Returns original data | 3 tests fail |
| **2D Horizontal Flip** | Reverse width only | Reverses all elements | 3 tests fail |
| **2D Cropping** | Proper H×W crops | 1D crops | 2 tests fail |
| **Padding support** | RandomCrop padding param | Param exists but unused | 1 test fails |

### TODO List from Implementation

1. **Line 219**: "TODO: Properly set shape metadata on returned tensor" (Reshape)
2. **Line 279**: "TODO: Implement proper 2D image resizing with interpolation" (Resize)
3. **Line 332**: "TODO: Implement proper 2D center cropping for image tensors" (CenterCrop)
4. **Line 387**: "TODO: Implement proper 2D random cropping for image tensors" (RandomCrop)
5. **Line 437**: "TODO: For proper image flipping, need to reverse only width dimension" (RandomHorizontalFlip)
6. **Line 484**: "TODO: Implement proper image rotation" (RandomRotation)
7. **Missing**: RandomErasing implementation
8. **Missing**: RandomVerticalFlip implementation

## Implementation Strategy

### Four-Phase Parallel + Sequential Approach

```text
Phase 1 (Engineer 1) ─┐
                       │
Phase 2 (Engineer 2) ─┼─→ Phase 4 (Engineer 4)
                       │   Test Integration
Phase 3 (Engineer 3) ─┘   & Verification
```

### Phase 1: Basic Transform Fixes (Engineer 1) - 4-6 hours

**Deliverables**:
1. Fix RandomHorizontalFlip for proper 2D flipping
2. Implement RandomRotation (simplified but functional)
3. Add Pipeline type (alias or wrapper for Compose)

**Tests to Enable**:
- `test_random_flip_always`
- `test_random_flip_never`
- `test_random_horizontal_flip_probability`
- `test_random_rotation_range`
- `test_random_rotation_no_change`
- `test_random_rotation_fill_value`
- `test_compose_random_augmentations`
- `test_augmentation_determinism_in_pipeline`

**Technical Details**:

RandomHorizontalFlip fix:
```mojo
# Current: Reverses all elements
for i in range(data.num_elements() - 1, -1, -1):
    flipped.append(Float32(data[i]))

# Fixed: Reverse only width dimension
var size = int(sqrt(float(data.num_elements())))
for row in range(size):
    for col in range(size - 1, -1, -1):  # Reverse columns
        var idx = row * size + col
        flipped.append(Float32(data[idx]))
```

RandomRotation implementation:
```mojo
# Calculate rotation angle
var angle_range = self.degrees[1] - self.degrees[0]
var rand_val = float(random_si64(0, 1000000)) / 1000000.0
var angle = self.degrees[0] + (rand_val * angle_range)

# Implement simplified rotation (nearest neighbor)
# Full affine transformation with interpolation is complex
# Start with 90-degree increments or nearest-neighbor sampling
```

Pipeline type:
```mojo
# Option 1: Simple alias
alias Pipeline = Compose

# Option 2: Wrapper struct (more explicit)
@value
struct Pipeline(Transform):
    var compose: Compose
    fn __init__(out self, owned transforms: List[Transform]):
        self.compose = Compose(transforms^)
    fn __call__(self, data: Tensor) raises -> Tensor:
        return self.compose(data)
```

### Phase 2: Cropping Transforms (Engineer 2) - 3-4 hours

**Deliverables**:
1. Fix RandomCrop for proper 2D cropping
2. Add padding support to RandomCrop
3. Fix CenterCrop for proper 2D cropping

**Tests to Enable**:
- `test_random_crop_varies_location`
- `test_random_crop_with_padding`

**Technical Details**:

RandomCrop fix:
```mojo
# Assume square tensor (tests use this)
var size = int(sqrt(float(num_elements)))
var crop_h = self.size[0]
var crop_w = self.size[1]

# Random position
var max_h = size - crop_h
var max_w = size - crop_w
var top = int(random_si64(0, max_h + 1))
var left = int(random_si64(0, max_w + 1))

# Extract 2D crop
for h in range(crop_h):
    for w in range(crop_w):
        var idx = (top + h) * size + (left + w)
        cropped.append(Float32(data[idx]))
```

Padding support:
```mojo
# If padding specified, pad tensor first
if self.padding:
    var pad = self.padding.value()
    var padded_size = size + 2 * pad
    var padded = List[Float32](capacity=padded_size * padded_size)

    # Pad with zeros (or configurable value)
    for h in range(padded_size):
        for w in range(padded_size):
            if h < pad or h >= size + pad or w < pad or w >= size + pad:
                padded.append(0.0)
            else:
                var orig_idx = (h - pad) * size + (w - pad)
                padded.append(Float32(data[orig_idx]))

    # Use padded data for cropping
    data = Tensor(padded^)
    size = padded_size
```

CenterCrop fix:
```mojo
var size = int(sqrt(float(num_elements)))
var crop_h = self.size[0]
var crop_w = self.size[1]

# Center position
var top = (size - crop_h) // 2
var left = (size - crop_w) // 2

# Extract center crop (2D)
for h in range(crop_h):
    for w in range(crop_w):
        var idx = (top + h) * size + (left + w)
        cropped.append(Float32(data[idx]))
```

### Phase 3: New Transforms (Engineer 3) - 4-5 hours

**Deliverables**:
1. Implement RandomErasing from scratch
2. Implement RandomVerticalFlip from scratch

**Tests to Enable**:
- `test_random_erasing_basic`
- `test_random_erasing_scale`

**Technical Details**:

RandomErasing implementation:
```mojo
@value
struct RandomErasing(Transform):
    var p: Float64
    var scale: Tuple[Float64, Float64]
    var fill_value: Float64

    fn __init__(
        out self,
        p: Float64 = 0.5,
        scale: Tuple[Float64, Float64] = (0.02, 0.33),
        fill_value: Float64 = 0.0
    ):
        self.p = p
        self.scale = scale
        self.fill_value = fill_value

    fn __call__(self, data: Tensor) raises -> Tensor:
        # Probability check
        var rand_val = float(random_si64(0, 1000000)) / 1000000.0
        if rand_val >= self.p:
            return data

        # Determine erase region size
        var scale_range = self.scale[1] - self.scale[0]
        var scale_rand = float(random_si64(0, 1000000)) / 1000000.0
        var erase_scale = self.scale[0] + (scale_rand * scale_range)

        var num_elements = data.num_elements()
        var erase_area = int(float(num_elements) * erase_scale)
        var size = int(sqrt(float(num_elements)))
        var erase_size = int(sqrt(float(erase_area)))

        # Random position
        var max_pos = size - erase_size
        var top = int(random_si64(0, max_pos + 1))
        var left = int(random_si64(0, max_pos + 1))

        # Erase rectangular region
        var result = List[Float32](capacity=num_elements)
        for h in range(size):
            for w in range(size):
                var idx = h * size + w
                var in_erase = (
                    h >= top and h < top + erase_size and
                    w >= left and w < left + erase_size
                )
                if in_erase:
                    result.append(Float32(self.fill_value))
                else:
                    result.append(Float32(data[idx]))

        return Tensor(result^)
```

RandomVerticalFlip implementation:
```mojo
@value
struct RandomVerticalFlip(Transform):
    var p: Float64

    fn __init__(out self, p: Float64 = 0.5):
        self.p = p

    fn __call__(self, data: Tensor) raises -> Tensor:
        # Probability check
        var rand_val = float(random_si64(0, 1000000)) / 1000000.0
        if rand_val >= self.p:
            return data

        # Flip vertically (reverse rows)
        var size = int(sqrt(float(data.num_elements())))
        var flipped = List[Float32](capacity=data.num_elements())

        for row in range(size - 1, -1, -1):
            for col in range(size):
                var idx = row * size + col
                flipped.append(Float32(data[idx]))

        return Tensor(flipped^)
```

### Phase 4: Test Integration (Engineer 4) - 2-3 hours

**Dependencies**: Phases 1, 2, 3 must complete first

**Deliverables**:
1. Uncomment all 14 test functions
2. Fix any compilation errors
3. Fix any assertion failures
4. Verify all tests pass
5. Run full test suite to check for regressions

**Process**:

1. **Uncomment tests one-by-one**:
   - Start with simpler tests (e.g., `test_random_rotation_no_change`)
   - Verify each compiles before moving to next
   - Document any issues found

2. **Run each test**:
   ```bash
   mojo test tests/shared/data/transforms/test_augmentations.mojo
   ```

3. **Fix failures**:
   - Check test output for specific assertion failures
   - Coordinate with implementation engineers if fixes needed
   - Document workarounds or limitations

4. **Verify determinism**:
   - Run deterministic tests multiple times
   - Ensure same seed produces same results
   - Verify randomness tests actually vary

5. **Check for regressions**:
   - Run full test suite across all modules
   - Verify no existing functionality broken
   - Check memory usage and performance

## Technical Challenges and Solutions

### Challenge 1: Tensor Shape Metadata
**Problem**: Mojo's Tensor doesn't expose H, W, C dimensions
**Solution**: Assume square tensors, infer size with `sqrt(num_elements())`
**Limitation**: Only works for square images in tests
**Long-term**: Wait for Tensor API improvements or create wrapper

### Challenge 2: Image Rotation Complexity
**Problem**: Proper rotation needs affine transforms + interpolation
**Solution**: Implement simplified rotation (nearest neighbor)
**Limitation**: Lower quality than production implementations
**Long-term**: Implement full affine transformation module

### Challenge 3: Random Number Generation
**Problem**: Need consistent RNG with seed control
**Solution**: Use `TestFixtures.set_seed()` for deterministic tests
**Limitation**: Global seed affects all subsequent random operations
**Long-term**: Implement RNG with state management per transform

### Challenge 4: Memory Management
**Problem**: Creating new tensors for each transform is inefficient
**Solution**: Accept overhead for correctness first, optimize later
**Limitation**: May have performance issues with large pipelines
**Long-term**: Implement in-place transforms where possible

## Testing Strategy

### Deterministic Tests (with seed)
- `test_random_augmentation_deterministic`
- `test_augmentation_determinism_in_pipeline`

**Verification**: Run multiple times, results must be identical

### Randomness Tests (without seed)
- `test_random_augmentation_varies`
- `test_random_crop_varies_location`

**Verification**: Multiple runs produce different results

### Probability Tests
- `test_random_horizontal_flip_probability` (p=0.5 → ~50% flip rate)
- `test_random_flip_always` (p=1.0 → always flip)
- `test_random_flip_never` (p=0.0 → never flip)

**Verification**: Statistical checks over many iterations

### Edge Case Tests
- `test_random_rotation_no_change` (degrees=0 → no change)
- `test_random_rotation_fill_value` (fill value used correctly)

**Verification**: Boundary conditions handled correctly

## Success Criteria

### Implementation Complete When:
- [ ] All 7 TODOs resolved
- [ ] RandomErasing implemented and tested
- [ ] RandomVerticalFlip implemented and tested
- [ ] Pipeline type available
- [ ] All transforms work with 2D images

### Tests Complete When:
- [ ] All 14 test functions uncommented
- [ ] All 14 tests pass consistently
- [ ] Deterministic tests produce identical results
- [ ] Randomness tests produce varying results
- [ ] No regressions in existing test suites

### Integration Complete When:
- [ ] Full test suite passes
- [ ] Code review approved
- [ ] Documentation updated
- [ ] Ready for Package phase (issue #411)

## Timeline Estimate

| Phase | Engineer | Duration | Dependencies |
|-------|----------|----------|--------------|
| Phase 1 | Engineer 1 | 4-6 hours | None |
| Phase 2 | Engineer 2 | 3-4 hours | None |
| Phase 3 | Engineer 3 | 4-5 hours | None |
| Phase 4 | Engineer 4 | 2-3 hours | Phases 1-3 |
| **Total** | **4 engineers** | **13-18 hours** | **Parallel + Sequential** |

**Critical Path**: Phases 1-3 parallel (4-6 hours) → Phase 4 sequential (2-3 hours) = **6-9 hours total**

## Communication Protocol

### Before Starting
- All engineers read this plan and issue #409 (test expectations)
- Engineers identify any blockers or questions
- Coordinate on any shared code or interfaces

### During Implementation
- Engineers post updates in issue comments
- Any API changes discussed before implementing
- Blockers escalated to Implementation Specialist immediately

### Integration Point
- Engineers 1-3 notify when complete
- Engineer 4 begins integration only after all dependencies met
- Any issues found fed back to responsible engineer

### After Completion
- Engineer 4 posts verification report
- Implementation Specialist reviews all code
- Prepare for PR creation and review

## Next Steps

1. **Implementation Specialist**: Review and approve this plan
2. **Engineers 1-3**: Begin parallel implementation
3. **Engineer 4**: Prepare test environment and monitoring
4. **All**: Daily standup updates on progress

## References

- Issue #408: [Plan] Image Augmentations
- Issue #409: [Test] Image Augmentations
- Issue #410: [Impl] Image Augmentations (this issue)
- Test file: `/home/user/ml-odyssey/tests/shared/data/transforms/test_augmentations.mojo`
- Implementation: `/home/user/ml-odyssey/shared/data/transforms.mojo`
- Test utilities: `/home/user/ml-odyssey/tests/shared/conftest.mojo`
