# Phase 2: Memory Safety Design for ExTensor

**Date**: 2025-11-22
**Issues**: #1904-#1908 (MOJO-001 to MOJO-005)
**Priority**: P0 CRITICAL - Blocks production use

---

## Current Problems

### 1. Use-After-Free (MOJO-001)

**Location**: `reshape()` and `slice()` methods

```mojo
fn reshape(self, new_shape: List[Int]) raises -> ExTensor:
    var result = ExTensor(new_shape, self._dtype)
    result._data.free()  # Free allocated memory
    result._data = self._data  # Point to parent's data
    result._is_view = True
    return result^  # View escapes, but parent may be freed!
```

**Problem**: View shares parent's data pointer but has no lifetime guarantees. When parent is freed, view has dangling pointer.

**Example**:
```mojo
fn create_view() -> ExTensor:
    var parent = zeros(List[Int](10, 10), DType.float32)
    var view = parent.reshape(List[Int](100))
    return view^  # Parent freed here, view now has dangling pointer!
```

### 2. Double-Free (MOJO-002)

**Location**: `__del__()` destructor

```mojo
fn __del__(deinit self):
    if not self._is_view:
        self._data.free()  # Fragile! What if _is_view is corrupted?
```

**Problem**: Both parent and view can try to free the same memory.

### 3. Missing Lifetime Tracking (MOJO-003)

**Problem**: No mechanism to ensure parent outlives views. Need reference counting.

### 4. Memory Leak in Error Path (MOJO-004)

**Location**: `reshape()` line 315-322

**Problem**: If exception occurs after `ExTensor()` allocation but before `free()`, memory leaks.

### 5. Missing __copyinit__ (MOJO-005)

**Problem**: No explicit copy constructor. Mojo may synthesize shallow copy causing double-free.

---

## Design Solution: Reference Counting

Implement Arc-like (Atomic Reference Counted) pattern:

```mojo
struct ExTensor(Copyable, Movable):
    var _data: UnsafePointer[UInt8]
    var _shape: List[Int]
    var _strides: List[Int]
    var _dtype: DType
    var _numel: Int
    var _is_view: Bool

    # NEW: Reference counting
    var _refcount: UnsafePointer[Int]  # Shared counter

    fn __init__(out self, shape: List[Int], dtype: DType) raises:
        # ... existing allocation ...

        # NEW: Allocate and initialize refcount
        self._refcount = UnsafePointer[Int].alloc(1)
        self._refcount[] = 1  # Start with 1 reference

    fn __copyinit__(out self, existing: Self):
        """Copy constructor - increments reference count."""
        # Shallow copy all fields
        self._data = existing._data
        self._shape = existing._shape
        self._strides = existing._strides
        self._dtype = existing._dtype
        self._numel = existing._numel
        self._is_view = existing._is_view
        self._refcount = existing._refcount

        # NEW: Increment reference count
        if not self._is_view:
            self._refcount[] += 1

    fn __del__(deinit self):
        """Destructor - decrements ref count, frees if last reference."""
        if not self._is_view and self._refcount:
            self._refcount[] -= 1

            # If last reference, free everything
            if self._refcount[] == 0:
                self._data.free()
                self._refcount.free()
```

---

## Implementation Plan

### Phase 2.1: Add __copyinit__ (MOJO-005) - QUICK WIN

**Priority**: Implement first (1 hour)
**Benefit**: Prevents accidental shallow copies immediately

```mojo
fn __copyinit__(out self, existing: Self):
    """Deep copy constructor with reference counting.

    Creates a new reference to the same underlying data.
    Increments the reference count to track shared ownership.
    """
    # Shallow copy all metadata
    self._data = existing._data
    self._shape = existing._shape
    self._strides = existing._strides
    self._dtype = existing._dtype
    self._numel = existing._numel
    self._is_view = existing._is_view
    self._refcount = existing._refcount  # Share the counter

    # Increment reference count (shared ownership)
    if not self._is_view and self._refcount:
        self._refcount[] += 1
```

### Phase 2.2: Add Reference Counting (MOJO-003) - FOUNDATION

**Priority**: Implement second (2-3 hours)
**Benefit**: Enables safe shared ownership

**Changes**:
1. Add `_refcount: UnsafePointer[Int]` field to struct
2. Allocate refcount in `__init__()`, initialize to 1
3. Update `__copyinit__()` to increment refcount
4. Update `__del__()` to decrement and free if zero

### Phase 2.3: Fix reshape/slice (MOJO-001) - USE-AFTER-FREE

**Priority**: Implement third (1-2 hours)
**Benefit**: Views can safely share parent's data

**Changes**:
```mojo
fn reshape(self, new_shape: List[Int]) raises -> ExTensor:
    # Validate shape
    var new_numel = 1
    for i in range(len(new_shape)):
        new_numel *= new_shape[i]

    if new_numel != self._numel:
        raise Error("Cannot reshape: element count mismatch")

    # Create view by copying parent and updating shape
    var result = self  # Uses __copyinit__, increments refcount!
    result._shape = new_shape
    # Recalculate strides for new shape
    result._strides = calculate_strides(new_shape)

    return result^
```

### Phase 2.4: Fix double-free (MOJO-002) - SAFETY

**Priority**: Fixed by Phase 2.2 (refcounting)
**Benefit**: Reference counting prevents double-free automatically

### Phase 2.5: Fix memory leaks (MOJO-004) - CLEANUP

**Priority**: Implement last (1 hour)
**Benefit**: Clean error paths

**Solution**: Avoid intermediate allocation in reshape/slice (already fixed by 2.3)

---

## Testing Strategy

### 1. Basic Reference Counting Tests
```mojo
fn test_copy_increments_refcount():
    var a = zeros(List[Int](10), DType.float32)
    var b = a  # __copyinit__ increments refcount
    # Both a and b share data, refcount = 2
    # When a destroyed, refcount = 1, data not freed
    # When b destroyed, refcount = 0, data freed
```

### 2. View Lifetime Tests
```mojo
fn test_view_outlives_parent():
    var view: ExTensor
    {
        var parent = zeros(List[Int](100), DType.float32)
        view = parent.reshape(List[Int](10, 10))
        # parent destroyed here, but refcount = 2, data not freed
    }
    # view still valid, refcount = 1
    var val = view[0, 0]  # Should work!
```

### 3. Memory Leak Tests
```mojo
fn test_no_leaks_on_exception():
    try:
        var t = zeros(List[Int](10), DType.float32)
        var reshaped = t.reshape(List[Int](5, 2))  # Should not leak
    except:
        pass  # Exception handling
```

---

## Estimated Effort

| Task | Effort | Risk |
|------|--------|------|
| Phase 2.1: __copyinit__ | 1 hour | LOW - straightforward |
| Phase 2.2: Reference counting | 2-3 hours | MEDIUM - pointer arithmetic |
| Phase 2.3: Fix reshape/slice | 1-2 hours | LOW - simplified by refcount |
| Phase 2.4: Double-free | 0 hours | - (fixed by 2.2) |
| Phase 2.5: Memory leaks | 1 hour | LOW - code cleanup |
| **Testing** | 2-3 hours | MEDIUM - need thorough tests |
| **TOTAL** | **7-10 hours** | **MEDIUM** |

**Note**: Original estimate was 1-2 weeks. With systematic approach and existing FIXME guidance, can complete in 1-2 days.

---

## Success Criteria

- [ ] `__copyinit__` implemented and tested
- [ ] Reference counting working (copies increment, deletes decrement)
- [ ] Views can outlive parents without use-after-free
- [ ] No double-free crashes
- [ ] No memory leaks (Valgrind clean)
- [ ] All existing tests still pass
- [ ] New tests verify reference counting behavior

---

## References

- GitHub Issues: #1904 (MOJO-001), #1905 (MOJO-002), #1906 (MOJO-003), #1907 (MOJO-004), #1908 (MOJO-005)
- Summary Issue: #1920
- FIXME comments in: `shared/core/extensor.mojo` lines 76-95, 162-170, 262-270, 317-319, 328-330
