# Configure Gitattributes

## Overview
Create and configure the .gitattributes file to specify how Git should handle different file types. This includes line ending normalization, diff settings for binary files, and marking files for Git LFS.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- Repository root directory exists
- Understanding of file types in the project
- Knowledge of Git LFS patterns

## Outputs
- .gitattributes file at repository root
- Line ending configurations for text files
- Binary file handling specified
- LFS patterns for large files configured

## Steps
1. Create .gitattributes file at repository root
2. Configure line ending normalization for text files
3. Mark binary files appropriately
4. Specify patterns for Git LFS tracking
5. Add comments explaining configurations

## Success Criteria
- [ ] .gitattributes file exists and is properly configured
- [ ] Text files have consistent line endings
- [ ] Binary files are handled correctly
- [ ] Large file patterns are specified for LFS

## Notes
.gitattributes ensures consistent file handling across different operating systems and properly configures Git LFS for large files like model weights.
