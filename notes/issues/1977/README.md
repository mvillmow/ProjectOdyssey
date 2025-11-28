# Issue #1977: Fix unsafe List initialization patterns

## Objective

Replace unsafe `List[Int]()` initialization patterns followed by immediate `.append()` calls in 4 locations across training modules. This was causing undefined behavior and crashes in the Training: Loops & Metrics test group.

## Deliverables

Fixed files:
- `shared/training/trainer_interface.mojo` (lines 297-304, 2 locations)
- `shared/training/metrics/accuracy.mojo` (line 112-113, 1 location)
- `shared/training/metrics/confusion_matrix.mojo` (line 325-326, 1 location)

## Success Criteria

- [x] All 4 unsafe List initialization patterns replaced with safe alternatives
- [x] trainer_interface.mojo batch_data_shape fixed (2-element list)
- [x] trainer_interface.mojo batch_labels_shape fixed (1-element list)
- [x] accuracy.mojo result_shape fixed (1-element list)
- [x] confusion_matrix.mojo result_shape fixed (1-element list)

## Implementation Details

### Pattern Fixed

**UNSAFE** (causes undefined behavior):
```mojo
var shape = List[Int]()
shape.append(size)
```

**SAFE** (proper initialization with capacity):
```mojo
var shape = List[Int](capacity)
shape[index] = value
```

### Changes Made

#### 1. trainer_interface.mojo (lines 297-304)

Fixed two List initializations in the `DataLoader.next()` method:

**batch_data_shape** (2-element list):
- Changed: `List[Int]()` with two appends
- To: `List[Int](2)` with direct assignment

**batch_labels_shape** (1-element list):
- Changed: `List[Int]()` with one append
- To: `List[Int](1)` with direct assignment
- Also fixed: `shape()[0]` → `shape()[1]` to correctly get feature dimension

#### 2. accuracy.mojo (line 112-113)

Fixed List initialization in the `argmax()` function:

**result_shape** (1-element list):
- Changed: `List[Int]()` with one append
- To: `List[Int](1)` with direct assignment

#### 3. confusion_matrix.mojo (line 325-326)

Fixed List initialization in the `argmax()` helper function:

**result_shape** (1-element list):
- Changed: `List[Int]()` with one append
- To: `List[Int](1)` with direct assignment

## References

- Issue: #1977
- Related test failures: Training: Loops & Metrics test group
- Language documentation: [Mojo Manual - Types](https://docs.modular.com/mojo/manual/types/)

## Notes

The root cause was using empty List initialization without pre-allocating capacity, then immediately calling append(). This caused undefined behavior because:

1. `List[Int]()` creates an empty list with zero capacity
2. Calling `append()` on an uninitialized list can access invalid memory
3. The fix allocates proper capacity upfront and uses indexed assignment instead

This is a safe pattern that:
- Pre-allocates exact capacity needed
- Uses direct array indexing (safe after initialization)
- Avoids dynamic growth during append operations
