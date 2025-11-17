# Base Dataset

## Overview

Define the foundational dataset interface for consistent data access across all implementations. This includes the dataset interface specification, length method for dataset size, and getitem method for indexed access. A well-defined interface enables interchangeable dataset implementations.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

- [01-dataset-interface/plan.md](01-dataset-interface/plan.md)
- [02-dataset-length/plan.md](02-dataset-length/plan.md)
- [03-dataset-getitem/plan.md](03-dataset-getitem/plan.md)

## Inputs

- Data source specifications
- Access pattern requirements
- Index and slicing needs

## Outputs

- Dataset interface defining required methods
- Length implementation for dataset size queries
- Getitem implementation for indexed data access

## Steps

1. Define dataset interface with core methods
2. Implement length method for size tracking
3. Create getitem method for data retrieval by index

## Success Criteria

- [ ] Dataset interface is clear and minimal
- [ ] Length method returns correct dataset size
- [ ] Getitem method provides indexed access to samples
- [ ] All child plans are completed successfully

## Notes

Keep the interface minimal and Pythonic. Follow standard Python sequence protocols. Ensure the interface works for various data types (images, text, tabular data).
