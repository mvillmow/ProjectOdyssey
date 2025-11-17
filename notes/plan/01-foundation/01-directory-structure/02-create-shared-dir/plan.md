# Create Shared Directory

## Overview

Set up the shared directory structure which contains reusable components that can be used across multiple paper implementations. Create subdirectories for core functionality, training utilities, data processing, and general utilities.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

- [01-create-core/plan.md](01-create-core/plan.md)
- [02-create-training/plan.md](02-create-training/plan.md)
- [03-create-data/plan.md](03-create-data/plan.md)
- [04-create-utils/plan.md](04-create-utils/plan.md)

## Inputs

- Repository root directory exists
- Understanding of common components across papers
- Knowledge of what code should be shared vs paper-specific

## Outputs

- shared/ directory at repository root
- shared/core/ for fundamental building blocks
- shared/training/ for training utilities
- shared/data/ for data processing
- shared/utils/ for general utilities
- README files in each subdirectory

## Steps

1. Create core subdirectory for fundamental components
2. Create training subdirectory for training-related utilities
3. Create data subdirectory for data processing code
4. Create utils subdirectory for general helper functions

## Success Criteria

- [ ] shared/ directory exists at repository root
- [ ] All subdirectories are created with proper structure
- [ ] Each subdirectory has a README explaining its purpose
- [ ] Structure supports code reuse across papers

## Notes

The shared directory is for code that will be used by multiple paper implementations. Keep a clear separation between what goes in shared vs what stays in individual papers.
