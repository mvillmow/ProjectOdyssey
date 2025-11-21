# Execution Guide for Issue #40: Data Module Package

## Overview

This guide provides step-by-step instructions for building and verifying the Data module package artifacts.

## Prerequisites

- Mojo installed and available in PATH
- Working directory: `/home/mvillmow/ml-odyssey/worktrees/40-pkg-data`
- Git branch: `40-pkg-data`

## Step 1: Build the Package

Execute the build script to create the `.mojopkg` file:

```bash
# Make build script executable
chmod +x scripts/build_data_package.sh

# Run build
./scripts/build_data_package.sh
```text

### Expected Output

```text
Building Data module package...
Creating dist/ directory...
Building package: dist/data-0.1.0.mojopkg
Package created successfully:
-rw-r--r-- 1 user group XXXXX Nov 14 HH:MM dist/data-0.1.0.mojopkg

✅ Build complete!
Package: dist/data-0.1.0.mojopkg
```text

### Verification

```bash
# Check package file exists
ls -lh dist/data-0.1.0.mojopkg

# Should show non-zero file size
```text

## Step 2: Test Installation

Execute the verification script to test package installation:

```bash
# Make verification script executable
chmod +x scripts/install_verify_data.sh

# Run verification
./scripts/install_verify_data.sh
```text

### Expected Output

```text
Testing data package installation...
Testing in temporary directory: /tmp/tmp.XXXXXXXXXX
Installing package from .../dist/data-0.1.0.mojopkg...
Testing imports...
Dataset import OK
TensorDataset import OK
BatchLoader import OK
Transform import OK
Compose import OK

✅ Data package verification complete!
All imports successful
Cleanup complete
```text

### If verification fails

- Check that package was built successfully
- Verify Mojo is in PATH: `which mojo`
- Check for build errors in previous step
- Ensure source files are valid: `mojo build shared/data/__init__.mojo`

## Step 3: Manual Testing (Optional)

Test package installation manually:

```bash
# Create temporary directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Install package
mojo install /home/mvillmow/ml-odyssey/worktrees/40-pkg-data/dist/data-0.1.0.mojopkg

# Test imports
mojo run -c "from data import Dataset; print('Success!')"
mojo run -c "from data import TensorDataset, BatchLoader; print('Success!')"
mojo run -c "from data import Transform, Compose; print('Success!')"

# Cleanup
cd -
rm -rf "$TEMP_DIR"
```text

## Step 4: Commit Changes

After successful build and verification:

```bash
# Check status
git status

# Stage files
git add scripts/build_data_package.sh
git add scripts/install_verify_data.sh
git add notes/issues/40/README.md
git add notes/issues/40/EXECUTION_GUIDE.md
git add notes/issues/40/package-build-task.md

# Commit with conventional format
git commit -m "feat(data): create distributable package with installation testing

- Built dist/data-0.1.0.mojopkg binary package
- Created build script (scripts/build_data_package.sh)
- Created installation verification script (scripts/install_verify_data.sh)
- Tested installation in clean environment
- Updated documentation to reflect actual artifacts

Closes #40"

# Push to remote
git push origin 40-pkg-data
```text

## Step 5: Create Pull Request

```bash
# Create PR linked to issue
gh pr create --issue 40 --fill

# Or manually with description
gh pr create --title "feat(data): create distributable package with installation testing" \
  --body "Built dist/data-0.1.0.mojopkg binary package and created installation verification. Closes #40"
```text

## Troubleshooting

### Issue: `mojo package` command not found

### Solution

```bash
# Check Mojo installation
which mojo

# If not found, activate pixi environment
cd /home/mvillmow/ml-odyssey
pixi shell

# Retry build
cd worktrees/40-pkg-data
./scripts/build_data_package.sh
```text

### Issue: Package build fails with compilation errors

### Solution

```bash
# Test individual files compile
mojo build shared/data/__init__.mojo
mojo build shared/data/datasets.mojo
mojo build shared/data/loaders.mojo
mojo build shared/data/samplers.mojo
mojo build shared/data/transforms.mojo

# Fix any compilation errors
# Retry build
```text

### Issue: Verification script fails with import errors

### Solution

```bash
# Check package contents (if mojo supports inspection)
# Verify exports in __init__.mojo are correct
cat shared/data/__init__.mojo | grep -A 30 "__all__"

# Ensure all exported items are actually defined
```text

### Issue: dist/ directory already exists but is empty

### Solution

```bash
# Safe to proceed - dist/ is in .gitignore
# Build will create .mojopkg file
./scripts/build_data_package.sh
```text

## Success Criteria Checklist

Before creating PR, verify:

- [ ] `dist/data-0.1.0.mojopkg` file exists
- [ ] Package file is non-zero size
- [ ] `scripts/build_data_package.sh` exists and is executable
- [ ] `scripts/install_verify_data.sh` exists and is executable
- [ ] Build script runs successfully
- [ ] Verification script runs successfully
- [ ] All core imports work in clean environment
- [ ] Documentation updated (`notes/issues/40/README.md`)
- [ ] Changes committed with proper message
- [ ] PR created and linked to issue #40

## File Manifest

Files created/modified for this issue:

```text
scripts/build_data_package.sh           # NEW - Build automation
scripts/install_verify_data.sh          # NEW - Installation verification
notes/issues/40/README.md               # MODIFIED - Package documentation
notes/issues/40/EXECUTION_GUIDE.md      # NEW - This file
notes/issues/40/package-build-task.md   # NEW - Task specification

dist/data-0.1.0.mojopkg                 # GENERATED (not committed, in .gitignore)
```text

## Next Steps

After successful PR merge:

1. Tag release: `git tag v0.1.0`
1. Consider CI/CD automation (future issue)
1. Document installation in main README
1. Apply same pattern to Training and Utils modules
