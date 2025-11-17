# Detect Paper Changes

## Overview

Identify which paper implementations were added or modified in a pull request to determine which papers need validation.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- [To be determined]

## Outputs

- Completed detect paper changes

## Steps

1. [To be determined]

## Success Criteria

- [ ] Correctly identifies all modified papers
- [ ] Handles new papers being added
- [ ] Handles existing papers being updated
- [ ] Ignores non-paper directory changes
- [ ] Outputs paper list in usable format

## Notes

Use git diff-tree or actions/checkout with fetch-depth to get full diff. Parse paths like papers/lenet-5/implementation.mojo to extract "lenet-5" as paper name.
