# Dataset Getitem

## Overview
Implement the __getitem__ method for datasets to retrieve individual samples by index. This is the core data access method that enables both random access and iteration. Proper getitem implementation is essential for all dataset operations.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- Index (integer) or slice for accessing data
- Data storage (files, memory, database)
- Transform/preprocessing pipeline
- Error handling for invalid indices

## Outputs
- __getitem__ method returning sample(s)
- Support for integer indexing and slicing
- Consistent return format (tuple of data, label)
- Clear errors for out-of-bounds access

## Steps
1. Implement integer indexing for single samples
2. Add slice support for range access
3. Apply transforms/preprocessing to retrieved data
4. Handle edge cases (negative indices, out of bounds)

## Success Criteria
- [ ] Returns correct sample for valid indices
- [ ] Supports both positive and negative indexing
- [ ] Slice notation works correctly
- [ ] Clear errors for invalid indices

## Notes
Return format should be consistent: (data, label) tuple or similar structure. Apply transforms after retrieval. Support negative indexing like Python lists. Ensure thread-safety if needed.
