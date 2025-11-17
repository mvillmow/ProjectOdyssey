# Setup Git LFS

## Overview

Set up Git Large File Storage (LFS) for handling large files like trained models, checkpoints, and datasets. Configure LFS tracking patterns and ensure the repository is ready to handle large ML artifacts efficiently.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Repository root directory exists
- .gitattributes file exists
- Understanding of which files need LFS
- Git LFS installed on system

## Outputs

- Git LFS initialized in repository
- .gitattributes updated with LFS patterns
- Documentation on LFS usage
- Instructions for contributors

## Steps

1. Initialize Git LFS in the repository
2. Configure LFS tracking for model files (.pt, .pth, .onnx, etc.)
3. Configure LFS tracking for dataset files
4. Configure LFS tracking for checkpoint files
5. Document LFS setup for contributors

## Success Criteria

- [ ] Git LFS is initialized and configured
- [ ] Large file patterns are tracked by LFS
- [ ] .gitattributes includes LFS tracking rules
- [ ] Documentation explains LFS usage to contributors

## Notes

Git LFS is essential for ML projects to avoid bloating the repository with large model files. Configure it early and document clearly for contributors.
