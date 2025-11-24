# Issue #651: [Plan] Setup Git LFS - Design and Documentation

## Objective

Set up Git Large File Storage (LFS) for handling large files like trained models, checkpoints, and datasets. This planning phase defines the specifications, architecture, and approach for configuring LFS tracking patterns to ensure the repository is ready to handle large ML artifacts efficiently.

## Deliverables

- Git LFS initialized in repository
- .gitattributes updated with LFS patterns for ML-specific file types
- Documentation on LFS usage for contributors
- Instructions for setting up and using Git LFS in the project

## Success Criteria

- [ ] Git LFS is initialized and configured
- [ ] Large file patterns are tracked by LFS
- [ ] .gitattributes includes LFS tracking rules
- [ ] Documentation explains LFS usage to contributors

## Design Decisions

### Architecture Choice: Git LFS vs Repository Split

**Decision**: Use Git LFS for managing large ML artifacts within the main repository.

### Rationale

- ML projects regularly produce large binary files (models, checkpoints, datasets)
- Git LFS allows version control of large files without bloating repository size
- Keeps project structure unified (code + artifacts) for easier workflow
- Industry standard for ML/AI repositories (PyTorch, TensorFlow, Hugging Face)

### Alternatives Considered

1. **Separate artifact repository**: Would increase complexity for contributors and CI/CD
1. **Cloud storage only**: Would lose version tracking and reproducibility
1. **No large file management**: Would cause repository bloat and performance issues

### File Type Categories for LFS Tracking

**Decision**: Track three categories of large files with Git LFS:

1. **Model Files**: Trained model weights and architectures
   - Extensions: `.pt`, `.pth`, `.onnx`, `.pb`, `.h5`, `.keras`, `.safetensors`
   - Rationale: Core ML artifacts that can be 100MB-10GB+

1. **Dataset Files**: Training and validation datasets
   - Extensions: `.npz`, `.npy`, `.parquet`, `.arrow`, `.feather`
   - Compressed archives: `.tar.gz`, `.zip` (in data directories)
   - Rationale: Large datasets that benefit from versioning

1. **Checkpoint Files**: Training snapshots and intermediate states
   - Extensions: `.ckpt`, `.checkpoint`
   - Directories: `checkpoints/`, `snapshots/`
   - Rationale: Training artifacts that can accumulate quickly

### Why These Categories

- Covers all major ML artifact types in the project
- Aligns with common ML framework conventions
- Prevents accidental commits of large files
- Supports reproducibility and experiment tracking

### .gitattributes Configuration Strategy

**Decision**: Use explicit pattern matching in .gitattributes with comments for each category.

### Format

```text
# Git LFS - Model Files
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
...

# Git LFS - Dataset Files
*.npz filter=lfs diff=lfs merge=lfs -text
...

# Git LFS - Checkpoints
*.ckpt filter=lfs diff=lfs merge=lfs -text
...
```text

### Rationale

- Explicit patterns prevent surprises and make tracking rules obvious
- Comments improve maintainability and understanding
- Consistent with existing .gitattributes format in repository
- Makes it easy to add/remove patterns as project evolves

### Alternatives Considered

1. **Directory-based tracking** (e.g., `models/**`): Less flexible, harder to maintain
1. **Minimal patterns**: Risk missing important file types
1. **Wildcard patterns**: Could catch unintended files

### Documentation Requirements

**Decision**: Create three levels of documentation:

1. **Contributor Setup Guide**: How to install and configure Git LFS
   - Location: `docs/git-lfs-setup.md` or section in contributing guide
   - Audience: New contributors

1. **Usage Instructions**: Working with LFS-tracked files
   - Location: Main README.md or developer guide
   - Audience: All developers

1. **Inline Comments**: Explain tracking rules in .gitattributes
   - Location: .gitattributes file itself
   - Audience: Advanced users modifying configuration

### Rationale

- Multi-level documentation serves different user needs
- Setup guide reduces friction for new contributors
- Usage instructions prevent common mistakes
- Inline comments improve maintainability

### Integration with Existing Git Configuration

**Decision**: Extend existing .gitattributes file rather than replacing it.

### Current .gitattributes

```text
# SCM syntax highlighting & preventing 3-way merges
pixi.lock merge=binary linguist-language=YAML linguist-generated=true
```text

### Approach

- Add LFS patterns as a new section below existing rules
- Preserve existing configuration for pixi.lock
- Use section headers for organization

### Rationale

- Maintains existing repository behavior
- Clear separation between LFS and non-LFS rules
- Follows principle of minimal changes

## References

- **Source Plan**: [notes/plan/01-foundation/02-configuration-files/03-git-config/03-setup-git-lfs/plan.md](notes/plan/01-foundation/02-configuration-files/03-git-config/03-setup-git-lfs/plan.md)
- **Parent Plan**: [notes/plan/01-foundation/02-configuration-files/03-git-config/plan.md](notes/plan/01-foundation/02-configuration-files/03-git-config/plan.md)
- **Related Issues**:
  - #652: [Test] Setup Git LFS - Write Tests
  - #653: [Impl] Setup Git LFS - Implementation
  - #654: [Package] Setup Git LFS - Integration and Packaging
  - #655: [Cleanup] Setup Git LFS - Refactor and Finalize

### External Resources

- [Git LFS Official Documentation](https://git-lfs.github.com/)
- [GitHub: Managing large files with Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files)
- [Git LFS Best Practices for ML Projects](https://dvc.org/doc/user-guide/data-management/large-dataset-optimization)

## Implementation Notes

(This section will be filled during implementation phases with findings, decisions, and any deviations from the plan)
