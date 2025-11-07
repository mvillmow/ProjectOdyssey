# Dataset Length

## Overview
Implement the __len__ method for datasets to return the total number of samples. This enables size queries, progress tracking, and proper batch calculation in data loaders. Length is a fundamental property for dataset manipulation.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- Data source with known or computable size
- Indexing strategy for counting samples
- Handle for dynamic or streaming datasets

## Outputs
- __len__ method returning integer count
- Correct count for all dataset types
- Support for both static and dynamic sizing
- Clear behavior for infinite datasets

## Steps
1. Implement __len__ for static datasets
2. Handle dynamic datasets with lazy computation
3. Define behavior for infinite/streaming datasets
4. Ensure consistency with __getitem__ indexing

## Success Criteria
- [ ] __len__ returns correct sample count
- [ ] Works with Python's len() builtin
- [ ] Consistent with actual indexable range
- [ ] Clear error for unsized datasets

## Notes
For file-based datasets, may need to scan or cache size. For in-memory datasets, return array length. For infinite datasets, consider raising error or returning special value.
