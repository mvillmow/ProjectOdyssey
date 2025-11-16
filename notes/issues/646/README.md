# Issue #646: [Plan] Configure Gitattributes - Design and Documentation

## Objective

Create and configure the .gitattributes file to specify how Git should handle different file types in the ML Odyssey
repository. This includes line ending normalization for cross-platform consistency, diff settings for binary files, and
marking files for Git LFS tracking. The .gitattributes file ensures consistent file handling across different operating
systems and properly configures Git LFS for large files like model weights.

## Deliverables

- .gitattributes file at repository root
- Line ending configurations for text files (Python, Mojo, YAML, JSON, Markdown, etc.)
- Binary file handling specified (images, model files, executables)
- LFS patterns for large files configured (model weights, checkpoints, datasets)
- Inline documentation explaining each configuration

## Success Criteria

- [ ] .gitattributes file exists at repository root
- [ ] Text files have consistent line ending normalization (LF on checkout, platform-native on working directory)
- [ ] Binary files are marked correctly to prevent text-based diff/merge attempts
- [ ] Large file patterns are specified for Git LFS tracking
- [ ] File includes clear comments explaining each section
- [ ] Configuration covers all relevant file types for ML projects (Mojo, Python, models, datasets)

## Design Decisions

### Line Ending Normalization Strategy

**Decision**: Use `text=auto` as the default, with specific overrides for file types that require explicit handling.

**Rationale**:

- `text=auto` lets Git automatically detect text files and normalize line endings
- Ensures LF in repository, platform-appropriate in working directory
- Prevents CRLF/LF conflicts across Windows, Linux, and macOS development environments
- Explicit overrides for shell scripts, Python, Mojo to guarantee LF

**Alternatives Considered**:

- `* text=auto eol=lf` (force LF everywhere) - Rejected: Can cause issues with Windows tools expecting CRLF
- No normalization - Rejected: Would result in inconsistent line endings across platforms
- Per-file-type only (no default) - Rejected: Too verbose and easy to miss new file types

### Binary File Handling

**Decision**: Explicitly mark binary files with `binary` attribute to prevent text-based operations.

**Rationale**:

- Prevents Git from attempting text diffs on binary files (images, PDFs, compiled files)
- Avoids corruption from line ending normalization
- Improves performance by skipping text analysis
- Makes merge conflicts clearer (binary files can't be auto-merged)

**File Types to Mark**:

- Images: `*.png`, `*.jpg`, `*.jpeg`, `*.gif`, `*.ico`, `*.svg`
- Documents: `*.pdf`, `*.docx`
- Archives: `*.zip`, `*.tar.gz`, `*.7z`
- Compiled: `*.so`, `*.dylib`, `*.dll`, `*.exe`
- ML artifacts: `*.npy`, `*.npz` (NumPy arrays)

### Git LFS Pattern Strategy

**Decision**: Configure LFS patterns in .gitattributes using `filter=lfs` attribute.

**Rationale**:

- Centralizes LFS configuration in .gitattributes (single source of truth)
- Makes LFS tracking visible in version control
- Ensures consistent LFS behavior across all contributors
- Supports both `git lfs track` command output and manual editing

**Patterns to Include**:

- Model weights: `*.pt`, `*.pth`, `*.onnx`, `*.pb`, `*.h5`, `*.safetensors`
- Checkpoints: `*.ckpt`, `*.checkpoint`
- Datasets: `*.parquet`, `*.feather`, `*.arrow`
- Large binaries: Files >100MB should be tracked

**Integration with Setup Git LFS**:

- This component creates the .gitattributes structure
- Issue #651 (Setup Git LFS) will run `git lfs install` and add specific track patterns
- The two components work together: .gitattributes defines patterns, LFS setup initializes tracking

### linguist Configuration

**Decision**: Include linguist directives for accurate language statistics and syntax highlighting.

**Rationale**:

- `linguist-language` helps GitHub understand file types (e.g., `pixi.lock` is YAML)
- `linguist-generated=true` excludes generated/vendor files from stats
- `linguist-vendored=true` marks third-party code
- Improves repository insights and code browsing experience

**Current Example**:

```gitattributes
pixi.lock merge=binary linguist-language=YAML linguist-generated=true
```

### File Organization

**Decision**: Organize .gitattributes into logical sections with clear comments.

**Sections**:

1. Auto-detection and defaults
2. Text files (language-specific)
3. Binary files
4. Git LFS patterns
5. SCM-specific configurations (linguist, merge strategies)

**Rationale**:

- Makes file easier to maintain and understand
- Clear separation of concerns
- Easy to locate and update specific patterns
- Follows industry best practices (similar to .gitignore organization)

### Cross-Platform Considerations

**Decision**: Ensure configurations work identically on Windows, Linux, and macOS.

**Key Aspects**:

- Use forward slashes in patterns (Git standard)
- Avoid platform-specific paths or assumptions
- Test line ending normalization on all platforms
- Document any platform-specific behavior

## References

### Source Plan

[notes/plan/01-foundation/02-configuration-files/03-git-config/02-configure-gitattributes/plan.md](/home/mvillmow/ml-odyssey-manual/notes/plan/01-foundation/02-configuration-files/03-git-config/02-configure-gitattributes/plan.md)

### Parent Plan

[notes/plan/01-foundation/02-configuration-files/03-git-config/plan.md](/home/mvillmow/ml-odyssey-manual/notes/plan/01-foundation/02-configuration-files/03-git-config/plan.md)

### Related Issues

- Issue #645: [Plan] Update Gitignore - Design and Documentation (sibling)
- Issue #651: [Plan] Setup Git LFS - Design and Documentation (sibling, depends on .gitattributes)
- Issue #647: [Test] Configure Gitattributes - Write Tests
- Issue #648: [Implementation] Configure Gitattributes - Build Functionality
- Issue #649: [Packaging] Configure Gitattributes - Integration
- Issue #650: [Cleanup] Configure Gitattributes - Refactor and Finalize

### External Documentation

- [Git Attributes Documentation](https://git-scm.com/docs/gitattributes)
- [Git LFS Specification](https://git-lfs.github.com/)
- [GitHub Linguist Documentation](https://github.com/github/linguist)
- [Line Ending Best Practices](https://www.git-scm.com/book/en/v2/Customizing-Git-Git-Configuration#_formatting_and_whitespace)

## Implementation Notes

(This section will be populated during the implementation phase)
