# Release Process

Guide to releasing new versions of ML Odyssey.

## Overview

ML Odyssey follows semantic versioning (MAJOR.MINOR.PATCH) with a structured release process to ensure quality
and stability.

## Versioning Scheme

**Semantic Versioning** (`MAJOR.MINOR.PATCH`):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

**Examples**:

- `1.0.0` → `2.0.0`: Breaking changes (remove deprecated API)
- `1.0.0` → `1.1.0`: New feature (add Attention layer)
- `1.0.0` → `1.0.1`: Bug fix (fix gradient computation)

## Release Cycle

**Regular releases**: Every 6-8 weeks

**Hotfix releases**: As needed for critical bugs

**Release types**:

- **Alpha** (`0.1.0-alpha.1`): Early development, unstable
- **Beta** (`0.1.0-beta.1`): Feature-complete, testing phase
- **RC** (`1.0.0-rc.1`): Release candidate, final testing
- **Stable** (`1.0.0`): Production-ready

## Release Checklist

### 1. Pre-Release (1-2 weeks before)

- [ ] Create release branch: `release/vX.Y.Z`
- [ ] Update version in `pyproject.toml` and relevant files
- [ ] Review and close related GitHub issues
- [ ] Update CHANGELOG.md with all changes since last release
- [ ] Run full test suite: `pixi run mojo test tests/`
- [ ] Run all benchmarks and compare with previous release
- [ ] Update documentation for new features
- [ ] Review and merge pending PRs

### 2. Release Candidate (1 week before)

- [ ] Tag RC: `git tag v1.0.0-rc.1`
- [ ] Build packages: `mojo package shared/`
- [ ] Deploy RC to staging
- [ ] Announce RC for community testing
- [ ] Monitor issue tracker for RC feedback
- [ ] Fix critical bugs found in RC
- [ ] Create new RC if needed: `v1.0.0-rc.2`

### 3. Final Release (Release day)

- [ ] Merge release branch to `main`
- [ ] Tag release: `git tag v1.0.0`
- [ ] Build final packages
- [ ] Create GitHub Release with release notes
- [ ] Upload packages to release
- [ ] Deploy documentation
- [ ] Announce release (blog, social media, mailing list)
- [ ] Update examples and tutorials
- [ ] Merge `main` back to `develop`

### 4. Post-Release (1-2 days after)

- [ ] Monitor issue tracker for release-related bugs
- [ ] Update download statistics
- [ ] Gather community feedback
- [ ] Plan hotfix if critical bugs found
- [ ] Start planning next release

## Creating a Release

### Step 1: Create Release Branch

```bash
```bash

# Create release branch from develop
git checkout develop
git pull origin develop
git checkout -b release/v1.2.0

# Update version number
sed -i 's/version = ".*"/version = "1.2.0"/' pyproject.toml

# Commit version bump
git add pyproject.toml
git commit -m "chore: bump version to 1.2.0"
git push -u origin release/v1.2.0

```text

### Step 2: Update CHANGELOG

```markdown
```markdown

# CHANGELOG.md

## [1.2.0] - 2026-01-15

### Added

- New Attention layer with multi-head support
- SIMD optimization for Conv2D
- Data augmentation transforms (flip, rotate, crop)

### Changed

- Improved SGD optimizer convergence
- Updated documentation with more examples

### Fixed

- Fixed gradient computation in BatchNorm
- Resolved memory leak in DataLoader

### Deprecated

- Old API for creating models (use new API)

### Removed

- Dropped support for Mojo < 0.25

```text

### Step 3: Run Quality Checks

```bash
```bash

# Run all tests
pixi run mojo test tests/
pytest tests/

# Run pre-commit hooks
pre-commit run --all-files

# Run benchmarks
mojo benchmarks/scripts/run_benchmarks.mojo

# Build documentation
mkdocs build

# Validate agent configs
python3 tests/agents/validate_configs.py .claude/agents/

```text

### Step 4: Create Release Candidate

```bash
```bash

# Tag RC
git tag v1.2.0-rc.1
git push origin v1.2.0-rc.1

# Build packages
mojo package shared/ -o ml-odyssey-v1.2.0-rc.1.mojopkg

# Create pre-release on GitHub
gh release create v1.2.0-rc.1 \
    --prerelease \
    --title "v1.2.0 RC1" \
    --notes "Release candidate for v1.2.0. Please test and report issues."

# Upload package
gh release upload v1.2.0-rc.1 ml-odyssey-v1.2.0-rc.1.mojopkg

```text

### Step 5: Final Release

```bash
```bash

# Merge to main
git checkout main
git merge release/v1.2.0
git push origin main

# Tag final release
git tag v1.2.0
git push origin v1.2.0

# Build final packages
mojo package shared/ -o ml-odyssey-v1.2.0.mojopkg

# Create GitHub Release
gh release create v1.2.0 \
    --title "ML Odyssey v1.2.0" \
    --notes-file RELEASE_NOTES.md \
    ml-odyssey-v1.2.0.mojopkg

# Deploy docs
mkdocs gh-deploy

```text

### Step 6: Announce Release

```markdown
```markdown

**Template for release announcement:**

# ML Odyssey v1.2.0 Released

We're excited to announce ML Odyssey v1.2.0, featuring:

- **New Attention Layer**: Multi-head attention for transformer models
- **Performance**: 2x faster Conv2D with SIMD optimization
- **Data Augmentation**: Built-in transforms for image preprocessing

## Installation

```bash
```bash

pixi install ml-odyssey==1.2.0

```text

## What's New

[Brief highlights with examples]

## Breaking Changes

[List any breaking changes]

## Migration Guide

[If needed]

## Thanks

Special thanks to all contributors!

Full changelog: <https://github.com/owner/ml-odyssey/blob/main/CHANGELOG.md>

```text

## Hotfix Releases

For critical bugs in production:

### Hotfix Process

```bash
```bash

# Create hotfix branch from main
git checkout main
git checkout -b hotfix/v1.2.1

# Fix the bug
[make fixes]

# Run targeted tests
pixi run mojo test tests/specific_test.mojo

# Bump patch version
sed -i 's/version = "1.2.0"/version = "1.2.1"/' pyproject.toml

# Commit and tag
git commit -am "fix: critical bug in gradient computation"
git tag v1.2.1

# Merge to main and develop
git checkout main
git merge hotfix/v1.2.1
git push origin main v1.2.1

git checkout develop
git merge hotfix/v1.2.1
git push origin develop

# Release immediately
gh release create v1.2.1 \
    --title "ML Odyssey v1.2.1 (Hotfix)" \
    --notes "Hotfix release: Fixed critical bug in gradient computation"

```text

## Release Artifacts

Each release should include:

- **Source code** (automatic via GitHub)
- **Mojo packages** (`.mojopkg` files)
- **Documentation** (deployed to GitHub Pages)
- **CHANGELOG.md** (included in release notes)
- **Migration guide** (if breaking changes)

## Deprecation Policy

**Deprecation process**:

1. **Announce**: Deprecate in MINOR release, keep functionality
2. **Warn**: Add deprecation warnings to code
3. **Document**: Update docs with migration guide
4. **Remove**: Remove in next MAJOR release

**Example**:

```mojo
```mojo

@deprecated("Use new_function() instead. Will be removed in v2.0.0")
fn old_function(x: Tensor) -> Tensor:
    return new_function(x)

```text

## Backward Compatibility

**Maintain compatibility**:

- PATCH releases: 100% backward compatible
- MINOR releases: Backward compatible, new features
- MAJOR releases: Breaking changes allowed

**Test compatibility**:

```bash
```bash

# Run tests against previous release
pixi install ml-odyssey==1.1.0
pytest tests/compatibility/

```text

## Release Tools

**GitHub CLI**:

```bash
```bash

# Create release
gh release create v1.2.0 --generate-notes

# List releases
gh release list

# View release
gh release view v1.2.0

```text

**Mojo Packaging**:

```bash
```bash

# Build package
mojo package shared/ -o shared.mojopkg

# Test package
mojo run -I shared.mojopkg examples/test.mojo

```text

## Related Documentation

- **Contributing Guide** (`CONTRIBUTING.md` in repo root) - Development workflow
- [CI/CD Pipeline](ci-cd.md) - Automated testing and deployment
- [Testing Strategy](../core/testing-strategy.md) - Quality assurance

## Summary

**Release Process**:

1. Create release branch
2. Update version and CHANGELOG
3. Run quality checks
4. Create RC for testing
5. Fix critical bugs
6. Tag and release
7. Announce and document

**Key Points**:

- Follow semantic versioning
- Test thoroughly before release
- Maintain backward compatibility
- Deprecate gracefully
- Document all changes
- Communicate clearly with users

**Next Steps**:

- Review current issues for next release
- Plan feature roadmap
- Set release date
- Assign release manager
