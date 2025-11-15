# Package Build Task for Issue #40

## Objective

Create actual distributable package artifacts for the Data module according to Package phase requirements.

## Context

- **Worktree**: `/home/mvillmow/ml-odyssey/worktrees/40-pkg-data`
- **Module**: `shared/data/`
- **Version**: `0.1.0` (from `__init__.mojo`)
- **Previous misunderstanding**: Package phase was incorrectly completed as documentation-only

## Required Deliverables

### 1. Build .mojopkg file

```bash
# Create dist directory
mkdir -p dist

# Build package
mojo package shared/data -o dist/data-0.1.0.mojopkg

# Verify package was created
ls -lh dist/data-0.1.0.mojopkg
file dist/data-0.1.0.mojopkg
```

### 2. Create installation verification script

File: `scripts/install_verify_data.sh`

```bash
#!/bin/bash
# Installation verification script for Data module package
# Usage: ./scripts/install_verify_data.sh

set -e

echo "Testing data package installation..."

# Get absolute path to package
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PACKAGE_PATH="$REPO_ROOT/dist/data-0.1.0.mojopkg"

if [ ! -f "$PACKAGE_PATH" ]; then
    echo "Error: Package not found at $PACKAGE_PATH"
    exit 1
fi

# Create temporary directory
TEMP_DIR=$(mktemp -d)
echo "Testing in temporary directory: $TEMP_DIR"

# Cleanup function
cleanup() {
    cd "$REPO_ROOT"
    rm -rf "$TEMP_DIR"
    echo "Cleanup complete"
}
trap cleanup EXIT

cd "$TEMP_DIR"

# Install package
echo "Installing package from $PACKAGE_PATH..."
mojo install "$PACKAGE_PATH"

# Test imports
echo "Testing imports..."
mojo run -c "from data import Dataset; print('Dataset import OK')"
mojo run -c "from data import TensorDataset; print('TensorDataset import OK')"
mojo run -c "from data import BatchLoader; print('BatchLoader import OK')"
mojo run -c "from data import Transform; print('Transform import OK')"
mojo run -c "from data import Compose; print('Compose import OK')"

echo ""
echo "✅ Data package verification complete!"
echo "All imports successful"
```

### 3. Update issue documentation

File: `notes/issues/40/README.md`

Update to reflect ACTUAL artifacts created (not just verification).

### 4. Commit changes

```bash
# Stage changes
git add dist/.gitignore  # To ensure dist/ directory is tracked properly
git add scripts/install_verify_data.sh
git add notes/issues/40/README.md

# Commit with conventional format
git commit -m "feat(data): create distributable package with installation testing

- Built dist/data-0.1.0.mojopkg binary package
- Created installation verification script
- Tested installation in clean environment
- Updated documentation to reflect actual artifacts

Closes #40"
```

## Execution Steps

1. Create `dist/` directory
2. Build `.mojopkg` file using `mojo package` command
3. Verify package file exists and is non-empty
4. Create `scripts/` directory if it doesn't exist
5. Create installation verification script
6. Make script executable (`chmod +x`)
7. Test the verification script
8. Update `notes/issues/40/README.md`
9. Stage and commit changes
10. Report results

## Success Criteria

- [ ] `dist/data-0.1.0.mojopkg` file exists
- [ ] Package file is non-empty (> 0 bytes)
- [ ] `scripts/install_verify_data.sh` exists and is executable
- [ ] Verification script runs successfully
- [ ] Issue documentation updated
- [ ] Changes committed with proper message

## Important Notes

- The `dist/` directory is in `.gitignore` - this is correct (artifacts should not be committed)
- However, we need to create a `.gitignore` entry IN the `dist/` directory to track it
- The package should include all modules: datasets, loaders, samplers, transforms
- Version 0.1.0 is specified in `shared/data/__init__.mojo`
