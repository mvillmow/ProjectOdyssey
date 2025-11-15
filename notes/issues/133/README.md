# Issue #133: [Plan] Configure Gitattributes - Design and Documentation

## Objective

Create and configure the .gitattributes file to specify how Git should handle different file types, including line
ending normalization, diff settings for binary files, and marking files for Git LFS.

## Deliverables

- Detailed specifications for .gitattributes patterns
- Text file normalization rules (line ending consistency across platforms)
- Binary file handling rules (proper detection and diff strategies)
- Language detection configuration (linguist settings)
- Diff and merge strategies for specific file types
- LFS patterns for large files (model weights, datasets)

## Success Criteria

- [ ] .gitattributes file exists and is properly configured
- [ ] Text files have consistent line endings
- [ ] Binary files are handled correctly
- [ ] Large file patterns are specified for LFS

## References

- Source Plan: `/notes/plan/01-foundation/02-configuration-files/03-git-config/02-configure-gitattributes/plan.md`
- Parent Component: #143 (Git Config)
- Related Issues: #134 (Test), #135 (Impl), #136 (Package), #137 (Cleanup)
- Git Documentation: [gitattributes](https://git-scm.com/docs/gitattributes)
- Git LFS Documentation: [Git LFS](https://git-lfs.github.com/)

## Implementation Notes

[To be filled during implementation]

## Design Decisions

### Line Ending Normalization

**Strategy**: Enforce LF line endings in repository, auto-convert on checkout per platform

**Rationale**:

- Consistent line endings across platforms (Windows/Linux/macOS)
- Prevents unnecessary diffs caused by CRLF/LF differences
- Standard practice for cross-platform development

**Implementation**:

```text
# Set default behavior - auto-normalize line endings
* text=auto

# Explicitly declare text files that should be normalized
*.py text eol=lf
*.mojo text eol=lf
*.🔥 text eol=lf
*.md text eol=lf
*.yml text eol=lf
*.yaml text eol=lf
*.json text eol=lf
*.toml text eol=lf
*.sh text eol=lf

# Windows-specific files that need CRLF
*.bat text eol=crlf
*.ps1 text eol=crlf
```

### Binary File Detection

**Strategy**: Explicitly mark binary files to prevent text normalization and enable appropriate diff handling

**Rationale**:

- Prevents corruption of binary files
- Enables specialized diff strategies (e.g., exiftool for images)
- Improves Git performance by skipping text diffs on binaries

**Implementation**:

```text
# Binary files - do not normalize
*.png binary
*.jpg binary
*.jpeg binary
*.gif binary
*.ico binary
*.pdf binary
*.zip binary
*.tar binary
*.gz binary
*.bz2 binary
*.7z binary

# Model files and weights
*.h5 binary
*.pkl binary
*.pth binary
*.onnx binary
*.pb binary
*.tflite binary

# Compiled files
*.pyc binary
*.so binary
*.dylib binary
*.dll binary
*.exe binary
```

### Language Statistics

**Strategy**: Use linguist attributes to control repository language statistics on GitHub

**Rationale**:

- Accurately represent project language composition
- Exclude vendored code, generated files, and documentation from stats
- Highlight Mojo as primary language

**Implementation**:

```text
# Documentation - exclude from language stats
*.md linguist-documentation

# Vendored code - mark as vendored
third_party/** linguist-vendored

# Generated files - mark as generated
*.gen.py linguist-generated
*_pb2.py linguist-generated
```

### Diff and Merge Strategies

**Strategy**: Configure specialized diff/merge handling for specific file types

**Rationale**:

- Better readability for structured data (JSON, YAML)
- Prevent merge conflicts in generated files
- Enable semantic diffs for images and other binary formats

**Implementation**:

```text
# Use specific diff drivers for certain file types
*.json diff=json
*.yaml diff=yaml
*.yml diff=yaml

# Prevent merging of generated files
*.gen.py merge=ours
*_pb2.py merge=ours

# Image diff using exiftool (if configured)
*.png diff=exif
*.jpg diff=exif
*.jpeg diff=exif
```

### Git LFS Configuration

**Strategy**: Track large binary files with Git LFS to keep repository size manageable

**Rationale**:

- Model weights and datasets can be hundreds of MB or GB
- Git LFS stores large files outside main repository
- Improves clone and fetch performance
- Essential for ML/AI projects with large assets

**Implementation**:

```text
# Large model files - track with LFS
*.h5 filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text

# Large datasets - track with LFS
*.csv filter=lfs diff=lfs merge=lfs -text
*.tsv filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text

# Compressed archives (if large datasets)
*.tar.gz filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
```

### Special Considerations

**Mojo Files**:

- Treat `.mojo` and `.🔥` files as text with LF endings
- Enable syntax highlighting via linguist if needed
- Ensure proper diff and merge handling

**Python Files**:

- Standard text normalization with LF
- Exclude generated protobuf files (`*_pb2.py`) from language stats
- Use merge=ours for generated files to prevent conflicts

**Configuration Files**:

- Normalize YAML/JSON/TOML with LF endings
- Use specialized diff drivers for better readability
- Document any exceptions (e.g., Windows-specific configs)

## Next Steps

After planning is approved (#133):

1. **Test Phase (#134)**: Write tests to verify .gitattributes behavior
2. **Implementation (#135)**: Create .gitattributes file with documented patterns
3. **Packaging (#136)**: Integrate with repository setup and documentation
4. **Cleanup (#137)**: Review and refine based on actual usage patterns
