# Release Process

Version management, release workflow, and changelog generation for ML Odyssey.

## Overview

ML Odyssey follows a structured release process with semantic versioning, automated checks, and comprehensive
release notes. This document covers the complete release workflow from planning to deployment.

## Versioning Strategy

### Semantic Versioning

ML Odyssey uses [Semantic Versioning 2.0.0](https://semver.org/):

```text
MAJOR.MINOR.PATCH

v1.2.3
│ │ │
│ │ └─ Patch: Bug fixes
│ └─── Minor: New features (backward compatible)
└───── Major: Breaking changes
```

**Examples**:

- `v0.1.0` → `v0.1.1`: Bug fix (increment patch)
- `v0.1.1` → `v0.2.0`: New feature (increment minor)
- `v0.2.0` → `v1.0.0`: Breaking change (increment major)

### Pre-Release Versions

For alpha/beta/rc releases:

```text
v1.0.0-alpha.1    # Alpha release
v1.0.0-beta.2     # Beta release
v1.0.0-rc.1       # Release candidate
v1.0.0            # Stable release
```

### Version Numbering

**Major version 0** (pre-1.0):

- Indicates unstable API
- Breaking changes allowed in minor versions
- Example: `v0.1.0`, `v0.2.0`, `v0.3.0`

**Major version 1+** (stable):

- Stable API
- Breaking changes only in major versions
- Example: `v1.0.0`, `v1.1.0`, `v2.0.0`

## Release Workflow

### Phase 1: Planning

#### Step 1: Create milestone

```bash
# Create GitHub milestone for release
gh milestone create "v0.2.0" --due-date 2024-12-31
```

#### Step 2: Assign issues

```bash
# Assign issues to milestone
gh issue edit 123 --milestone "v0.2.0"
```

#### Step 3: Create release branch

```bash
# For major/minor releases
git checkout -b release/v0.2.0 main

# For patches
git checkout -b release/v0.1.1 v0.1.0
```

### Phase 2: Development

#### Follow normal workflow

1. Create feature branches
2. Implement changes
3. Write tests
4. Submit PRs
5. Code review
6. Merge to release branch

**Version bump** in `pixi.toml`:

```toml
[project]
name = "ml-odyssey"
version = "0.2.0"  # Updated version
```

### Phase 3: Testing

#### Step 1: Run full test suite

```bash
# All tests
pixi run pytest tests/

# With coverage
pixi run pytest tests/ --cov=shared --cov=papers --cov-report=html

# Open coverage report
open htmlcov/index.html
```

#### Step 2: Smoke tests

```bash
# Test installation
pixi install --force-reinstall

# Test examples
pixi run mojo run examples/quickstart.mojo

# Test paper implementations
pixi run mojo run papers/lenet5/train.mojo
```

#### Step 3: Performance benchmarks

```bash
# Run benchmarks
pixi run mojo run benchmarks/run_all.mojo

# Compare with baseline
python scripts/compare_benchmarks.py \
    --current results/benchmarks.json \
    --baseline baselines/v0.1.0.json
```

### Phase 4: Documentation

#### Step 1: Update CHANGELOG.md

```markdown
# Changelog

## [0.2.0] - 2024-01-15

### Added

- Data augmentation transforms (RandomCrop, RandomFlip)
- Learning rate schedulers (StepLR, CosineAnnealing)
- Model checkpointing with callbacks
- Performance benchmarks suite

### Changed

- Improved SIMD optimization in Conv2D layer (2x faster)
- Refactored training loop for better error handling

### Fixed

- Fixed gradient accumulation bug in BatchLoader
- Resolved memory leak in Tensor.backward()

### Deprecated

- `old_function()` - Use `new_function()` instead

### Removed

- None

### Security

- None
```

#### Step 2: Update documentation

```bash
# Build documentation
pixi run mkdocs build

# Preview locally
pixi run mkdocs serve

# Check for broken links
pixi run pytest tests/foundation/docs/test_links.py
```

#### Step 3: Update migration guide (for breaking changes)

````markdown
# Migration Guide: v0.1.x → v0.2.0

## Breaking Changes

### Training API

**Before (v0.1.x)**:

```mojo
var trainer = Trainer(model, optimizer)
trainer.train(data, epochs=10)
```

**After (v0.2.0)**:

```mojo
var trainer = Trainer(model, optimizer, loss_fn)  # Added loss_fn
trainer.train(train_loader, val_loader, epochs=10)  # Separate loaders
```

## Deprecated Features

- `Trainer.fit()` - Use `Trainer.train()` instead
- Will be removed in v1.0.0

````

### Phase 5: Release

#### Step 1: Create release PR

```bash
# Create PR from release branch to main
gh pr create \
  --base main \
  --head release/v0.2.0 \
  --title "Release v0.2.0" \
  --body "$(cat CHANGELOG.md | sed -n '/## \[0.2.0\]/,/## \[/p' | head -n -1)"
```

#### Step 2: Review and approve

- Code review by maintainers
- All CI checks pass
- Documentation reviewed
- Changelog approved

#### Step 3: Merge release PR

```bash
gh pr merge --squash --delete-branch
```

#### Step 4: Create Git tag

```bash
# On main branch after merge
git checkout main
git pull

# Create annotated tag
git tag -a v0.2.0 -m "Release v0.2.0

- Data augmentation transforms
- Learning rate schedulers
- Model checkpointing
- Performance improvements
"

# Push tag
git push origin v0.2.0
```

#### Step 5: Create GitHub release

```bash
# Auto-generate from tag
gh release create v0.2.0 \
  --title "v0.2.0 - Data Augmentation and Schedulers" \
  --notes "$(cat CHANGELOG.md | sed -n '/## \[0.2.0\]/,/## \[/p' | head -n -1)" \
  --latest
```

### Phase 6: Post-Release

#### Step 1: Update main branch

```bash
# Bump version to next development version
vim pixi.toml  # Change to v0.3.0-dev

git commit -m "chore: bump version to v0.3.0-dev"
git push origin main
```

#### Step 2: Announce release

- GitHub Discussions announcement
- Update README.md with latest version
- Social media (if applicable)

#### Step 3: Monitor issues

- Watch for bug reports
- Address critical issues quickly
- Plan patch release if needed

## Changelog Guidelines

### Format

Follow [Keep a Changelog](https://keepachangelog.com/):

```markdown
## [Version] - YYYY-MM-DD

### Added

- New features

### Changed

- Changes to existing features

### Deprecated

- Soon-to-be removed features

### Removed

- Removed features

### Fixed

- Bug fixes

### Security

- Security fixes
```

### Writing Good Entries

**DO**:

```markdown
### Added

- Learning rate scheduler with cosine annealing (#123)
- Data augmentation: RandomCrop and RandomFlip (#124)
```

**DON'T**:

```markdown
### Added

- Stuff
- Various improvements
```

## Hotfix Releases

For critical bugs in stable releases:

### Step 1: Create hotfix branch

```bash
# Branch from tagged version
git checkout -b hotfix/v0.1.1 v0.1.0
```

### Step 2: Fix the bug

```bash
# Make minimal fix
git commit -m "fix: resolve memory leak in Tensor.backward()"
```

### Step 3: Test thoroughly

```bash
pixi run pytest tests/
```

### Step 4: Release

```bash
# Merge to main and tagged version branch
git checkout main
git merge hotfix/v0.1.1

# Create tag
git tag -a v0.1.1 -m "Hotfix v0.1.1: Memory leak fix"
git push origin v0.1.1

# Create GitHub release
gh release create v0.1.1 --title "v0.1.1 - Hotfix"
```

## Release Checklist

### Pre-Release

- [ ] All milestone issues closed
- [ ] All tests passing
- [ ] Benchmarks run (no regressions)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Migration guide (if breaking changes)
- [ ] Version bumped in pixi.toml

### Release

- [ ] Release branch created
- [ ] Release PR created
- [ ] Release PR reviewed and approved
- [ ] Release PR merged to main
- [ ] Git tag created
- [ ] GitHub release created
- [ ] Release notes published

### Post-Release

- [ ] Version bumped to next dev version
- [ ] Release announced
- [ ] Documentation deployed
- [ ] No critical issues reported

## Automation

### CI/CD Integration

Automated release workflow (`.github/workflows/release.yml`):

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run tests
        run: pixi run pytest tests/

      - name: Build documentation
        run: pixi run mkdocs build

      - name: Create GitHub Release
        uses: actions/create-release@v1
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          draft: false
          prerelease: false
```

### Version Bumping

Script to automate version updates:

```python
#!/usr/bin/env python3
"""Bump version in all files."""

import re
import sys
from pathlib import Path

def bump_version(current: str, bump_type: str) -> str:
    """Bump version number."""
    major, minor, patch = map(int, current.split('.'))

    if bump_type == 'major':
        return f"{major + 1}.0.0"
    elif bump_type == 'minor':
        return f"{major}.{minor + 1}.0"
    elif bump_type == 'patch':
        return f"{major}.{minor}.{patch + 1}"

def main():
    if len(sys.argv) != 2:
        print("Usage: bump_version.py [major|minor|patch]")
        sys.exit(1)

    bump_type = sys.argv[1]

    # Read current version
    pixi_toml = Path("pixi.toml").read_text()
    match = re.search(r'version = "(\d+\.\d+\.\d+)"', pixi_toml)
    current = match.group(1)

    # Calculate new version
    new_version = bump_version(current, bump_type)

    print(f"Bumping version: {current} → {new_version}")

    # Update pixi.toml
    pixi_toml = re.sub(
        r'version = "\d+\.\d+\.\d+"',
        f'version = "{new_version}"',
        pixi_toml
    )
    Path("pixi.toml").write_text(pixi_toml)

    print(f"✓ Updated pixi.toml")

if __name__ == "__main__":
    main()
```

Usage:

```bash
python scripts/bump_version.py minor  # 0.1.0 → 0.2.0
```

## Best Practices

### DO

- ✅ Follow semantic versioning strictly
- ✅ Write detailed changelogs
- ✅ Test thoroughly before release
- ✅ Announce breaking changes clearly
- ✅ Provide migration guides
- ✅ Tag all releases

### DON'T

- ❌ Release without testing
- ❌ Make breaking changes in patches
- ❌ Skip changelog entries
- ❌ Release with failing CI
- ❌ Modify released tags
- ❌ Release on Fridays (unless critical)

## Next Steps

- **[CI/CD](ci-cd.md)** - Automated testing and deployment
- **[Architecture](architecture.md)** - System design
- **[API Reference](api-reference.md)** - API versioning

## Related Documentation

- [Contributing Guide](https://github.com/mvillmow/ml-odyssey/blob/main/CONTRIBUTING.md) - Contribution workflow
- [Testing Strategy](../core/testing-strategy.md) - Release testing
- [Workflow](../core/workflow.md) - Development process
