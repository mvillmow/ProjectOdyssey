# Build Instructions for Utils Package

This document provides step-by-step instructions for building the Utils module distributable package.

## Prerequisites

- Mojo installed and available in PATH
- Bash shell (Linux/macOS/WSL)
- Write permissions in the project directory

## Quick Build (Recommended)

```bash
# Navigate to project root
cd /path/to/ml-odyssey

# Make scripts executable
chmod +x scripts/*.sh

# Run complete packaging workflow
./scripts/package_utils.sh
```

This will:

1. Create the `dist/` directory
2. Build `dist/utils-0.1.0.mojopkg`
3. Make installation verification script executable
4. Test package installation (optional)

## Manual Build Steps

If you prefer step-by-step control:

### Step 1: Create Distribution Directory

```bash
mkdir -p dist
```

### Step 2: Build the Package

```bash
mojo package shared/utils -o dist/utils-0.1.0.mojopkg
```

**Expected output**: "Package created successfully" (or similar)

### Step 3: Verify Package Was Created

```bash
ls -lh dist/utils-0.1.0.mojopkg
```

**Expected output**: File size information (typically several KB to MB)

### Step 4: Make Verification Script Executable

```bash
chmod +x scripts/install_verify_utils.sh
```

### Step 5: Test Installation (Optional)

```bash
./scripts/install_verify_utils.sh
```

**Note**: This creates a temporary directory, installs the package, tests imports, and cleans up.

## Build Scripts Reference

### scripts/package_utils.sh

**Complete packaging workflow** (build + test + reporting)

```bash
./scripts/package_utils.sh
```

Options:

- `SKIP_INSTALL_TEST=1 ./scripts/package_utils.sh` - Skip installation testing

### scripts/build_utils_package.sh

**Build-only script** (no testing)

```bash
./scripts/build_utils_package.sh
```

Use this when you only need to rebuild the package without testing.

### scripts/install_verify_utils.sh

**Installation verification only** (assumes package exists)

```bash
./scripts/install_verify_utils.sh
```

Use this to test an existing package without rebuilding.

## Troubleshooting

### Mojo not found

**Issue**: `mojo: command not found` or Mojo is not in PATH

**Solution**:

```bash
# Verify Mojo installation
which mojo
mojo --version

# If not found, ensure Mojo is installed via Pixi:
pixi run mojo --version

# Or activate the Pixi environment:
pixi shell
mojo --version
```

### Permission errors on dist/

**Issue**: Cannot create dist/ directory or write to it

**Solution**:

```bash
mkdir -p dist
chmod 755 dist
```

### "Permission denied" when running scripts

**Issue**: Scripts not executable

**Solution**:

```bash
chmod +x scripts/*.sh
```

### Import errors after installation

**Issue**: Package installs but imports fail

**Solution**:

```bash
# Verify package installation
mojo list-packages

# Check if utils package is listed
# If not, reinstall:
mojo install dist/utils-0.1.0.mojopkg

# Test basic import
mojo run -c "import utils; print('Success!')"
```

### Package build fails with syntax errors

**Issue**: Source code has compilation errors

**Solution**:

1. Test each module individually:

```bash
mojo shared/utils/logging.mojo
mojo shared/utils/config.mojo
mojo shared/utils/io.mojo
mojo shared/utils/visualization.mojo
mojo shared/utils/random.mojo
mojo shared/utils/profiling.mojo
```

1. Fix any syntax errors reported
2. Retry the package build

### Installation test fails

**Issue**: Package installs but imports fail

**Solution**:

1. Verify package structure:

```bash
# Check if package exists
ls -lh dist/utils-0.1.0.mojopkg

# Try manual installation
mojo install dist/utils-0.1.0.mojopkg

# Test basic import
mojo run -c "import utils; print('Success!')"
```

1. Check for dependency issues
2. Ensure `__init__.mojo` exports are correct

## Expected Artifacts

After successful build:

```text
dist/
└── utils-0.1.0.mojopkg      # Binary package file

scripts/
├── build_utils_package.sh   # Build-only script
├── install_verify_utils.sh  # Installation verification
└── package_utils.sh         # Complete workflow
```

**Note**: The `dist/` directory is in `.gitignore` and should NOT be committed to git.

## Next Steps

After building the package:

1. **Test installation**: Run `./scripts/install_verify_utils.sh`
2. **Verify imports**: Test that all 49 exported symbols work
3. **Document results**: Update `notes/issues/45/README.md` with build results
4. **Commit changes**: Commit scripts and documentation (NOT the .mojopkg file)
5. **Create PR**: Link to Issue #45

## Reference

- Package phase guide: `agents/guides/package-phase-guide.md`
- Issue documentation: `notes/issues/45/README.md`
- Source code: `shared/utils/`
