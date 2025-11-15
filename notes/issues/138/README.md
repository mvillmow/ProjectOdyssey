# Issue #138: [Plan] Setup Git LFS - Design and Documentation

## Objective

Set up Git Large File Storage (LFS) for handling large files like trained models, checkpoints, and datasets. Configure LFS tracking patterns and ensure the repository is ready to handle large ML artifacts efficiently.

## Deliverables

- Git LFS installation and configuration specifications
- Comprehensive track patterns for large files (model weights, datasets, checkpoints)
- .gitattributes LFS pattern integration strategy
- Repository migration strategy (if needed for existing large files)
- Best practices documentation for contributors
- LFS workflow documentation (clone, fetch, pull strategies)

## Success Criteria

- [ ] Git LFS is initialized and configured
- [ ] Large file patterns are tracked by LFS
- [ ] .gitattributes includes LFS tracking rules
- [ ] Documentation explains LFS usage to contributors
- [ ] Planning documentation includes comprehensive track patterns
- [ ] Migration strategy documented for existing large files
- [ ] Contributor onboarding documentation created

## References

- Source Plan: `/notes/plan/01-foundation/02-configuration-files/03-git-config/03-setup-git-lfs/plan.md`
- Parent Component: #143 (Git Config)
- Related Issues: #139 (Test), #140 (Impl), #141 (Package), #142 (Cleanup)
- Related: #133 (Plan Gitattributes) - LFS integrates with .gitattributes
- Git LFS Documentation: <https://git-lfs.github.com/>
- Git LFS Tutorial: <https://github.com/git-lfs/git-lfs/wiki/Tutorial>

## Implementation Notes

This planning phase will define the comprehensive Git LFS setup needed for an ML research project. Git LFS is essential to prevent repository bloat from large model files, checkpoints, and datasets. The implementation should be transparent to contributors while ensuring efficient storage and retrieval of large files.

## Design Decisions

### LFS Track Patterns

Git LFS will track the following file patterns, organized by category:

#### 1. Model Weights and Checkpoints

Large binary files that change frequently during training:

- **PyTorch Models**: `*.pt`, `*.pth`, `*.ckpt`
- **TensorFlow Models**: `*.h5`, `*.pb`, `*.tfevents`
- **ONNX Models**: `*.onnx`
- **Safetensors**: `*.safetensors`
- **Generic Checkpoints**: `*.checkpoint`
- **Model Binaries**: `*.bin` (when used for model storage)

Rationale: These files are typically multi-MB to multi-GB in size and change with each training run. LFS ensures efficient storage without bloating the repository.

#### 2. Pre-trained Model Files

Downloaded or shared pre-trained models:

- **Hugging Face Models**: Files in `models/` directories with extensions above
- **Model Archives**: `*.tar.gz`, `*.zip` (when containing models)
- **Weights Only**: `*.weights`, `*.model`

Rationale: Pre-trained models are often several GB and should be versioned but not stored directly in Git.

#### 3. Large Dataset Files

Raw and processed dataset files (with careful consideration):

- **CSV/TSV**: `*.csv` (only if large, > 10MB)
- **Parquet**: `*.parquet`, `*.pq`
- **HDF5**: `*.hdf5`, `*.h5` (data format)
- **NumPy**: `*.npy`, `*.npz` (large arrays)
- **Arrow**: `*.arrow`, `*.feather`
- **Pickle**: `*.pkl`, `*.pickle` (large serialized objects)

Rationale: Dataset files can be very large. Use specific patterns rather than blanket tracking to avoid tracking small test fixtures.

**Important**: Consider NOT tracking datasets with LFS if they're available for download. Document download scripts instead.

#### 4. Media Files (if used for datasets)

- **Images**: `*.jpg`, `*.jpeg`, `*.png` (only in dataset directories)
- **Audio**: `*.wav`, `*.mp3`, `*.flac`
- **Video**: `*.mp4`, `*.avi`, `*.mov`

Rationale: Media files for ML datasets can be large. However, consider external storage (S3, etc.) for very large media datasets.

#### 5. Compiled Artifacts (optional, consider .gitignore instead)

- **Mojo Packages**: `*.mojopkg`
- **Shared Libraries**: `*.so`, `*.dylib`, `*.dll`
- **Static Libraries**: `*.a`, `*.lib`

Rationale: These could be tracked with LFS if they need versioning, but typically should be in .gitignore instead.

### Installation Strategy

#### System Requirements

Git LFS must be installed on:

1. **Developer Machines**: All contributors must have Git LFS installed
2. **CI/CD Runners**: GitHub Actions runners have LFS pre-installed
3. **Minimum Version**: Git LFS 2.0+ (current is 3.x)

#### Installation Methods

**macOS**:

```bash
brew install git-lfs
git lfs install
```

**Linux (Debian/Ubuntu)**:

```bash
sudo apt-get install git-lfs
git lfs install
```

**Linux (Fedora/RHEL)**:

```bash
sudo dnf install git-lfs
git lfs install
```

**Windows**:

```bash
# Via Git for Windows installer (includes LFS)
# Or via Chocolatey
choco install git-lfs
git lfs install
```

#### Repository Initialization

```bash
# Initialize LFS in the repository (one-time setup)
cd /path/to/ml-odyssey
git lfs install

# Add track patterns to .gitattributes
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.ckpt"
# ... (see comprehensive patterns below)

# Verify tracking
git lfs track

# Commit .gitattributes
git add .gitattributes
git commit -m "feat(config): initialize Git LFS tracking"
```

### Integration with .gitattributes

Git LFS uses `.gitattributes` to define which files are tracked. The integration strategy:

#### File Structure

The `.gitattributes` file will have separate sections:

```text
# =============================================================================
# Git LFS Tracking Patterns
# =============================================================================

# Model Weights and Checkpoints
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text

# Large Dataset Files
*.parquet filter=lfs diff=lfs merge=lfs -text
*.pq filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text

# Media Files (dataset directories only - use specific paths)
datasets/**/*.jpg filter=lfs diff=lfs merge=lfs -text
datasets/**/*.png filter=lfs diff=lfs merge=lfs -text
data/**/*.wav filter=lfs diff=lfs merge=lfs -text

# =============================================================================
# Regular Gitattributes (from Issue #133)
# =============================================================================
# (Other non-LFS patterns here)
```

#### Attribute Explanation

- `filter=lfs`: Use LFS filter for clean/smudge operations
- `diff=lfs`: Use LFS diff driver (shows pointer file changes, not binary diffs)
- `merge=lfs`: Use LFS merge driver
- `-text`: Mark as binary (don't perform line ending normalization)

#### Coordination with Issue #133

Issue #133 creates the base `.gitattributes` file. This issue (138) will:

1. **Add LFS section**: Append LFS patterns to existing .gitattributes
2. **Maintain structure**: Keep sections clearly separated with headers
3. **Avoid conflicts**: Ensure no overlap with non-LFS patterns
4. **Document integration**: Explain how LFS patterns interact with other attributes

### Repository Migration Strategy

If the repository already contains large files before LFS setup:

#### Assessment Phase

1. **Identify large files**:

   ```bash
   # Find files > 1MB in Git history
   git rev-list --objects --all | \
     git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
     awk '/^blob/ {if ($3 > 1048576) print $3/1048576 " MB", $4}' | \
     sort -rn | head -20
   ```

2. **Check current repository size**:

   ```bash
   git count-objects -vH
   ```

#### Migration Options

**Option 1: Clean Slate (Recommended for new repositories)**

- Start fresh with LFS from the beginning
- No migration needed - just initialize LFS and add patterns
- Best for repositories without significant history

**Option 2: Migrate Existing Files (For repositories with large files)**

Use `git lfs migrate` to rewrite history:

```bash
# Migrate specific file types
git lfs migrate import --include="*.pt,*.pth,*.ckpt" --everything

# Or migrate specific paths
git lfs migrate import --include="models/**/*.h5" --everything

# Verify migration
git lfs ls-files
```

**Warning**: This rewrites Git history. All contributors must re-clone.

**Option 3: Hybrid Approach**

- Use LFS for new files going forward
- Leave historical files in Git (if repo is already large)
- Document that old files are not LFS-tracked
- Consider cleanup in future major version

#### Current Repository Status

Based on repository age and size:

- **Assessment needed**: Check if large files already exist
- **Recommendation**: Use Option 1 (clean slate) since this is early in the project
- **Future-proofing**: Set up LFS before adding large files

### LFS Workflow Documentation

Contributors need clear documentation on working with LFS:

#### Cloning with LFS

```bash
# Standard clone (downloads LFS files automatically)
git clone https://github.com/mvillmow/ml-odyssey.git
cd ml-odyssey

# LFS files are automatically pulled
git lfs pull
```

#### Fetching and Pulling

```bash
# Pull with LFS files
git pull

# Manually fetch LFS objects if needed
git lfs fetch
git lfs pull
```

#### Checking LFS Status

```bash
# List tracked LFS files
git lfs ls-files

# Check LFS tracking patterns
git lfs track

# Verify a file is tracked by LFS
git lfs ls-files | grep "filename"

# Show LFS storage usage
git lfs env
```

#### Adding New LFS Files

```bash
# Files matching tracked patterns are automatically handled
git add models/new_model.pt
git commit -m "feat(models): add new trained model"

# Verify it's tracked as LFS
git lfs ls-files | grep "new_model.pt"
```

#### Common Issues and Solutions

**Issue**: LFS files not downloading

```bash
# Solution: Explicitly pull LFS objects
git lfs pull
```

**Issue**: Running out of LFS bandwidth/storage

```bash
# Solution: Use selective fetch (only specific files)
git lfs fetch --include="models/specific_model.pt"
```

**Issue**: Large files committed without LFS

```bash
# Solution: Migrate to LFS
git lfs migrate import --include="path/to/file.ext" --ref=HEAD
```

### LFS Storage and Bandwidth Considerations

#### GitHub LFS Quotas

- **Free tier**: 1 GB storage, 1 GB/month bandwidth
- **Per user**: Additional storage/bandwidth available for purchase
- **Organization**: Higher limits available

#### Optimization Strategies

1. **Selective Tracking**: Only track truly large files (> 1MB)
2. **External Storage**: Consider S3/GCS for very large datasets
3. **Download Scripts**: Provide scripts to download datasets instead of storing in LFS
4. **Prune Old LFS Objects**: Periodically clean up old model versions

#### Bandwidth Management

```bash
# Fetch only recent LFS objects (shallow clone)
git lfs fetch --recent

# Exclude LFS files during clone (fetch later)
GIT_LFS_SKIP_SMUDGE=1 git clone <repo>

# Selectively fetch LFS files
git lfs fetch --include="models/production/*.pt"
```

### Best Practices Documentation

#### For Contributors

1. **Verify LFS Installation**: Run `git lfs install` before first clone
2. **Check File Size**: Files > 1MB should probably be LFS-tracked
3. **Verify Tracking**: Use `git lfs ls-files` to confirm files are tracked
4. **Don't Commit Secrets**: LFS files are still in version control
5. **Be Mindful of Bandwidth**: Don't add hundreds of GB of models

#### For Repository Maintainers

1. **Monitor LFS Usage**: Check GitHub LFS storage/bandwidth regularly
2. **Update Patterns**: Add new patterns as needed for new file types
3. **Document Exceptions**: If some large files shouldn't use LFS, document why
4. **Review Large Commits**: Set up CI to warn on commits with large non-LFS files
5. **Plan for Growth**: Consider external storage for very large datasets

### Testing Strategy

The testing phase (Issue #139) should verify:

1. **LFS Installation**: Automated check that LFS is installed
2. **Pattern Matching**: Test files are correctly tracked by LFS
3. **Clone/Pull**: Verify LFS files download correctly
4. **File Size**: Ensure tracked files are pointers, not full content
5. **CI Integration**: Verify LFS works in GitHub Actions

Test cases should include:

```bash
# Test 1: Verify LFS initialization
test -f .gitattributes && grep "filter=lfs" .gitattributes

# Test 2: Add a large test file and verify it's tracked
dd if=/dev/zero of=test_model.pt bs=1M count=5
git add test_model.pt
git lfs ls-files | grep "test_model.pt"

# Test 3: Verify pointer file size (should be ~100 bytes, not 5MB)
git cat-file -p HEAD:test_model.pt | wc -c
```

### Security Considerations

1. **LFS Files Are Public**: Files in public repos are accessible via LFS
2. **No Encryption**: LFS doesn't encrypt files - don't store secrets
3. **Access Control**: Private repos restrict LFS access to authorized users
4. **Audit Trail**: LFS files are versioned like regular files

### Documentation Deliverables

The following documentation will be created:

1. **README Section**: Add "Working with Large Files" section to main README
2. **CONTRIBUTING.md**: Add LFS setup instructions for new contributors
3. **Developer Guide**: Detailed LFS workflow documentation
4. **Troubleshooting**: Common LFS issues and solutions
5. **CI/CD Notes**: Document LFS usage in GitHub Actions

## Next Steps

Once this planning documentation is approved:

1. **Issue #139 (Test)**: Create tests to verify LFS installation and tracking
2. **Issue #140 (Implementation)**: Initialize LFS and add track patterns to .gitattributes
3. **Issue #141 (Packaging)**: Ensure LFS is properly integrated and documented
4. **Issue #142 (Cleanup)**: Review and refine patterns based on actual usage

## Open Questions

1. **Dataset Storage Strategy**: Should large datasets be stored in LFS or externally (S3/GCS)?
   - Recommendation: Start with LFS for small datasets, document download scripts for large ones

2. **Model Versioning**: How many versions of trained models should we keep in LFS?
   - Recommendation: Keep production models, prune experimental ones

3. **CI/CD Bandwidth**: Will GitHub Actions LFS bandwidth be sufficient?
   - Recommendation: Monitor usage, optimize CI to fetch only needed files

4. **Migration Needed**: Are there already large files in the repository?
   - Action: Run assessment query to check

5. **Track Patterns Scope**: Should we track compiled artifacts (*.so, *.dll) or just ignore them?
   - Recommendation: Start with .gitignore, migrate to LFS only if versioning is needed
