# Release Process

Guide for creating releases of ML Odyssey.

## Overview

ML Odyssey uses semantic versioning and automated releases via GitHub Actions.
When a version tag is pushed, the release workflow automatically:

1. Validates the version format
2. Builds packages
3. Runs tests
4. Creates a GitHub release with artifacts
5. Publishes Docker images

## Semantic Versioning

Versions follow the format `MAJOR.MINOR.PATCH`:

- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Pre-release versions use suffixes: `v0.2.0-alpha.1`, `v0.2.0-beta.1`, `v0.2.0-rc.1`

## Creating a Release

### Using the Version Bump Script

The easiest way to create a release:

```bash
# Bump patch version (0.1.0 -> 0.1.1)
./scripts/bump_version.sh patch

# Bump minor version (0.1.0 -> 0.2.0)
./scripts/bump_version.sh minor

# Bump major version (0.1.0 -> 1.0.0)
./scripts/bump_version.sh major
```

The script will:

1. Update the VERSION file
2. Update version in pixi.toml and pyproject.toml (if present)
3. Commit the changes
4. Create an annotated tag

Then push to trigger the release:

```bash
git push origin main --tags
```

### Manual Release

If you prefer manual control:

```bash
# 1. Update VERSION file
echo "0.2.0" > VERSION

# 2. Commit
git add VERSION
git commit -m "chore: bump version to 0.2.0"

# 3. Create tag
git tag -a v0.2.0 -m "Release v0.2.0"

# 4. Push
git push origin main
git push origin v0.2.0
```

### Manual Workflow Dispatch

You can also trigger a release manually from GitHub Actions:

1. Go to Actions > Release workflow
2. Click "Run workflow"
3. Enter the version (e.g., `v0.2.0`)
4. Optionally mark as pre-release
5. Click "Run workflow"

## Generating Changelogs

Generate a changelog from commit history:

```bash
# Generate changelog since last tag
python scripts/generate_changelog.py

# Generate for specific version
python scripts/generate_changelog.py v0.2.0

# Generate between two tags
python scripts/generate_changelog.py v0.2.0 v0.1.0

# Output to file
python scripts/generate_changelog.py --output CHANGELOG.md
```

The script categorizes commits by conventional commit type:

- `feat:` -> Features
- `fix:` -> Bug Fixes
- `perf:` -> Performance
- `docs:` -> Documentation
- `refactor:` -> Refactoring
- `test:` -> Testing
- `ci:` -> CI/CD
- `chore:` -> Maintenance

## Pre-Release Checklist

Before creating a release, verify:

- [ ] All tests passing (`just test`)
- [ ] Documentation updated
- [ ] No uncommitted changes
- [ ] On the main branch
- [ ] CI passing on main

```bash
# Quick verification
git status
just test
gh run list --limit 5
```

## Release Artifacts

Each release includes:

- **Source archives**: `.tar.gz` source distribution
- **Python wheels**: `.whl` packages (if applicable)
- **Mojo packages**: `.mojopkg` files (if applicable)
- **Checksums**: `checksums.txt` with SHA256 hashes
- **Build manifest**: `BUILD_MANIFEST.txt` with build details
- **SBOM**: Software Bill of Materials

## Docker Images

Releases automatically publish Docker images to GHCR:

```bash
# Pull specific version
docker pull ghcr.io/mvillmow/projectodyssey:v0.2.0

# Pull latest release
docker pull ghcr.io/mvillmow/projectodyssey:latest
```

## Hotfix Releases

For urgent fixes to a released version:

```bash
# Create hotfix branch from release tag
git checkout -b hotfix/v0.1.1 v0.1.0

# Make fixes
git commit -m "fix: critical bug fix"

# Bump patch version
./scripts/bump_version.sh patch

# Push branch and tag
git push origin hotfix/v0.1.1
git push origin v0.1.1

# Create PR to merge hotfix back to main
gh pr create --base main --head hotfix/v0.1.1
```

## Troubleshooting

### Release workflow failed

Check the workflow logs:

```bash
gh run list --workflow=release.yml
gh run view <run-id> --log
```

### Tag already exists

If the release failed partway through:

```bash
# Delete local tag
git tag -d v0.2.0

# Delete remote tag (if pushed)
git push origin --delete v0.2.0

# Try again
./scripts/bump_version.sh patch
```

### Version mismatch

Ensure VERSION file matches the tag:

```bash
cat VERSION
git describe --tags
```

## Release Notes Template

For significant releases, include:

```markdown
## Highlights

- Major feature 1
- Major feature 2

## Breaking Changes

- Description of breaking change
- Migration: old way -> new way

## New Features

- Feature 1 (#issue)
- Feature 2 (#issue)

## Bug Fixes

- Fix 1 (#issue)
- Fix 2 (#issue)

## Performance

- Improvement 1 (Nx speedup)

## Documentation

- New guide for X
- Updated API reference

## Contributors

Thanks to @contributor1, @contributor2!
```

## Related Files

- `VERSION` - Current version file
- `.github/workflows/release.yml` - Release automation workflow
- `scripts/generate_changelog.py` - Changelog generation script
- `scripts/bump_version.sh` - Version bump script
