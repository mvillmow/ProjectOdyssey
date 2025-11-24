# Phase 3: Data Integrity Fixes for ExTensor Quantization

## Objective

Fix 5 critical data integrity issues in ExTensor quantization and type conversion functions that cause silent data loss, incorrect dimensions, and unsafe memory access.

## Status

Analysis and design phase - Ready to implement

## Issues to Fix

### #1909 (DATA-001): Padding Data Loss in Quantization

**Problem**: `to_mxfp4()` and `to_nvfp4()` pad tensors to 32-element and 16-element boundaries respectively, but lose the original size metadata.

**Example**:

- Input: 33 elements
- Padded to: 64 elements (2 blocks × 32)
- After `from_mxfp4()`: Returns 64 elements (WRONG! Should be 33)
- Result: Silent tensor dimension corruption

**Impact**: ALL quantization workflows produce wrong tensor dimensions

**Root Cause**: Original size is not preserved when padding

**Solution Approach** (CHOSEN: SIMPLEST - Add metadata field):

- Add `_original_numel_quantized: Int` field to ExTensor struct (-1 for non-quantized tensors)
- Store original size in `to_mxfp4()` and `to_nvfp4()`
- Restore original size in `from_mxfp4()` and `from_nvfp4()`
- This is simpler than creating a new QuantizedTensor wrapper struct
- Maintains backwards compatibility

### #1910 (DATA-002): Missing Size Tracking in Dequantization

**Problem**: `from_mxfp4()` and `from_nvfp4()` cannot restore original dimensions because no metadata exists.

**Impact**: Depends entirely on DATA-001 fix

**Solution**: Use `_original_numel_quantized` field from DATA-001 to restore actual size

### #1911 (DATA-003): Unvalidated DType in Conversions

**Problem**: 13 conversion methods don't re-validate dtype before bitcast operations:

- `to_fp8()`, `to_bf8()`, `to_mxfp4()`, `to_nvfp4()`
- `to_int8()`, `to_int16()`, `to_int32()`, `to_int64()`
- `to_uint8()`, `to_uint16()`, `to_uint32()`, `to_uint64()`

**Risk**: Wrong dtype reaches bitcast → garbage data. Example: Int32 bits interpreted as Float32

**Solution**: Add defensive assertions before bitcast in each loop that re-validate the dtype matches expected types

### #1912 (DATA-004): Unsafe Bitcasts Without Bounds Checks

**Problem**: Direct pointer arithmetic without bounds validation can read uninitialized memory

**Risk**: Access beyond buffer boundaries

**Solution**: Add bounds checking assertions before bitcast operations:

```mojo
if i >= self._numel:
    raise Error("Index out of bounds during bitcast")
```

### #1913 (DATA-005): Document FP16→FP32 Conversion

**Problem**: FP16 inputs are silently converted to FP32 before quantization without documentation

**Impact**: Different precision path → different quantized results vs direct FP32

**Solution**: Add clear documentation in docstrings explaining:

1. FP16 values are converted to FP32 before quantization
2. This means precision may differ from direct FP32 quantization
3. Document the exact conversion path for reproducibility

## Implementation Plan

### Step 1: Add metadata field to ExTensor struct

- Add `_original_numel_quantized: Int = -1` field to struct
- Document that -1 means non-quantized tensor

### Step 2: Modify to_mxfp4() and to_nvfp4()

- Store original numel before padding
- Create padded tensor as before
- Set `_original_numel_quantized` field with original size

### Step 3: Modify from_mxfp4() and from_nvfp4()

- Check if `_original_numel_quantized` is set (not -1)
- If set, reshape result to original size instead of padded size
- If not set, use padded size (backwards compatibility)

### Step 4: Add dtype validation to all 13 conversion methods

- Add assertion before each bitcast checking dtype is valid
- Use pattern: `if i == 0 and (dtype validation failed) then raise Error`

### Step 5: Add bounds checking to all bitcasts

- Add check: `if i >= self._numel: raise Error`

### Step 6: Update docstrings for DATA-005

- Document FP16→FP32 conversion behavior in all affected methods
- Add note: "FP16 values are converted to FP32 before quantization"

### Step 7: Create comprehensive tests

- Round-trip quantization with non-aligned sizes (33, 17, 100 elements)
- Verify original size is restored
- Test dtype validation catches wrong types
- Test bounds checking prevents out-of-bounds access
- Test FP16 behavior

## Files to Modify

1. `/home/mvillmow/ml-odyssey/shared/core/extensor.mojo`
   - Add field to struct (line ~82)
   - Modify struct documentation (line ~57)
   - Update `to_mxfp4()` (line ~1464)
   - Update `from_mxfp4()` (line ~1599)
   - Update `to_nvfp4()` (line ~1664)
   - Update `from_nvfp4()` (line ~1782)
   - Update `to_fp8()`, `to_bf8()` (line ~646, ~1344)
   - Update all `to_intX()` and `to_uintX()` methods (lines ~765-1270)
   - Replace FIXME comments with implementation

2. `/home/mvillmow/ml-odyssey/tests/test_data_integrity.mojo` (NEW FILE)
   - Quantization round-trip tests with non-aligned sizes
   - Dtype validation tests
   - Bounds checking tests
   - FP16 behavior tests

## Success Criteria

- [ ] All 5 DATA issues resolved
- [ ] Tests pass for quantization round-trip with non-aligned sizes
- [ ] Dtype validation catches wrong types
- [ ] Bounds checking prevents out-of-bounds access
- [ ] FP16 behavior is documented and tested
- [ ] Backwards compatible (existing code works)
- [ ] All FIXME comments replaced with implementation

## Testing Strategy

### Test 1: Quantization Round-Trip with Non-Aligned Size

```mojo
var t = zeros(List[Int](33), DType.float32)
var encoded = t.to_mxfp4()
var decoded = encoded.from_mxfp4()
assert decoded.numel() == 33  # Not 64!
```

### Test 2: Dtype Validation

```mojo
var t = zeros(List[Int](10), DType.int32)
try:
    var result = t.to_mxfp4()
    assert false  # Should have raised
except:
    pass  # Expected
```

### Test 3: Bounds Checking

```mojo
# Create tensor and manually set _numel to larger value
# Verify bitcast catches out-of-bounds access
```

### Test 4: FP16 Behavior

```mojo
var fp16_tensor = zeros(List[Int](10), DType.float16)
var result = fp16_tensor.to_mxfp4()
# Verify conversion works with FP16 input
```

## Design Decisions

1. **Metadata storage**: Chose simple field (`_original_numel_quantized`) instead of new struct
   - Simpler implementation
   - Maintains backwards compatibility
   - No API changes needed

2. **Backwards compatibility**: Non-quantized tensors use -1 as sentinel
   - Existing code continues to work
   - New behavior only activates for quantized tensors

3. **Dtype validation**: Added at bitcast points
   - Defensive programming pattern
   - Catches bugs early before data corruption

4. **Bounds checking**: Added before all memory access
   - Prevents undefined behavior
   - Clear error messages

5. **FP16 documentation**: Added to docstrings
   - Users understand the conversion path
   - Reproducible results

## References

- Phase 1 (Syntax Blockers): commit a38bc3ae
- Phase 2 (Memory Safety): commit 1f055f43
- ExTensor implementation: `/home/mvillmow/ml-odyssey/shared/core/extensor.mojo`
