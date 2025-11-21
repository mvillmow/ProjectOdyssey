# Issue #863: [Package] Download Mojo - Integration and Packaging

## Objective

Integrate the Mojo download implementation with the installer workflow and ensure proper packaging for deployment. Verify end-to-end download functionality, configure for production use, and create distributable package artifacts that enable reliable binary downloads with platform detection, progress tracking, and integrity verification.

## Deliverables

- Integrated download module in Mojo installer pipeline
- Configuration files for download URLs and checksums
- Download verification and validation scripts
- Integration tests with complete installer workflow
- Package build scripts and distribution configuration
- Installation verification script for clean environments
- Comprehensive packaging documentation

## Success Criteria

- [ ] Downloads correct Mojo version in production environment
- [ ] Network error handling works gracefully in real conditions
- [ ] Progress tracking displays correctly in installer UI
- [ ] File integrity verification works with real checksums
- [ ] Integration with platform detection is seamless
- [ ] All dependencies are properly configured
- [ ] Package installs correctly in clean environments
- [ ] Binary package file (.mojopkg) builds successfully
- [ ] Installation verification confirms all exports work
- [ ] End-to-end workflow tested on all supported platforms

## Package Scope

This is a **packaging phase** issue in the 5-phase development workflow:

1. **Plan** - #860 (Design and specification complete)
1. **Test** - #861 (Write comprehensive test suite)
1. **Implementation** - #862 (Build the download module)
1. **Package** - #863 (THIS ISSUE - Create distributable artifacts)
1. **Cleanup** - #864 (Refactor and finalization)

### What "Package" Means

This phase creates actual distributable artifacts, not just documentation:

### Created

- Build scripts for creating distributable archives/packages
- Installation verification scripts
- Packaging documentation
- Configuration files for production deployment
- Clear separation between source code (committed) and built artifacts (excluded)

### NOT Created

- Documentation-only deliverables
- Verification that existing code is "ready"
- Notes about packaging being "complete" without artifacts

## References

- **Source Plan**: `/notes/plan/03-tooling/03-setup-scripts/01-mojo-installer/02-download-mojo/plan.md`
- **Planning Issue**: #860 [Plan] Download Mojo - Design and Documentation
- **Testing Issue**: #861 [Test] Download Mojo - Write Tests
- **Implementation Issue**: #862 [Impl] Download Mojo - Implementation
- **Cleanup Issue**: #864 [Cleanup] Download Mojo - Cleanup and Finalization
- **Related**: #855-859 [Detect Platform] - provides platform information
- **Package Phase Guide**: `/agents/guides/package-phase-guide.md`
- **5-Phase Workflow**: `/notes/review/README.md`

## Project Context

### Mojo Installation Workflow

This module is part of the three-phase Mojo installation process:

1. **Platform Detection** (#855-859) - Identify target system architecture
   - Detects OS (Linux, macOS, Windows)
   - Detects CPU architecture (x86_64, ARM64, etc.)
   - Detects Python version and environment
   - Output: Platform information tuple

1. **Download Mojo** (#860-864) - Fetch Mojo compiler binary
   - Uses platform information from phase 1
   - Constructs appropriate download URL
   - Downloads binary with progress tracking
   - Verifies checksum for security
   - Extracts files if needed
   - Output: Downloaded and verified Mojo binary

1. **Install Mojo** (future phase) - Configure system for Mojo
   - Uses downloaded binary from phase 2
   - Installs to system paths
   - Configures PATH and environment variables
   - Verifies installation success
   - Output: Mojo compiler ready to use

### Architecture Overview

```text
installer/
├── platform_detector.py    (Issues #855-859)
│   └── Outputs: PlatformInfo(os, arch, python_version)
│
├── mojo_downloader.py      (Issues #860-864)
│   ├── Inputs: PlatformInfo from platform_detector
│   ├── Outputs: DownloadResult(path, checksum_verified, logs)
│   └── Features:
│       - URL construction for platform
│       - Download with progress bar
│       - Checksum verification
│       - Error handling and retries
│
├── mojo_installer.py       (Future phase)
│   ├── Inputs: DownloadResult from mojo_downloader
│   └── Outputs: InstallationResult(success, installed_version)
│
├── __init__.py
└── config/
    └── download_urls.yaml  (Configuration data)
```text

## Implementation Notes

### 1. Integration with Platform Detector

The download module receives platform information from platform detection issues (#855-859):

```python
# Usage in installer
from platform_detector import detect_platform
from mojo_downloader import download_mojo

platform_info = detect_platform()
download_result = download_mojo(
    platform_info=platform_info,
    version="24.5.0",  # Or latest version
    destination="/tmp/mojo_download"
)

if download_result.success:
    print(f"Downloaded to: {download_result.path}")
    print(f"Checksum verified: {download_result.checksum_verified}")
else:
    print(f"Download failed: {download_result.error}")
```text

### 2. Configuration File Structure

**File**: `config/download_urls.yaml`

```yaml
# Download URL configuration for different platforms
versions:
  "24.5.0":
    linux-x86_64:
      url: "https://modular.com/download/mojo/releases/24.5.0/mojo-24.5.0-linux-x86_64.tar.gz"
      checksum: "sha256:abc123..."
      size_bytes: 456789012

    linux-aarch64:
      url: "https://modular.com/download/mojo/releases/24.5.0/mojo-24.5.0-linux-aarch64.tar.gz"
      checksum: "sha256:def456..."
      size_bytes: 456789012

    macos-intel:
      url: "https://modular.com/download/mojo/releases/24.5.0/mojo-24.5.0-macos-intel.tar.gz"
      checksum: "sha256:ghi789..."
      size_bytes: 456789012

    macos-arm64:
      url: "https://modular.com/download/mojo/releases/24.5.0/mojo-24.5.0-macos-arm64.tar.gz"
      checksum: "sha256:jkl012..."
      size_bytes: 456789012

# Download settings
download:
  timeout_seconds: 300  # 5 minutes
  max_retries: 3
  retry_delay_seconds: 5
  chunk_size_bytes: 8192  # For progress bar
  verify_checksum: true
```text

### 3. Package Structure

```text
mojo_installer/
├── __init__.py                 # Package exports
├── platform_detector.py        # From #855-859
├── mojo_downloader.py          # Core download implementation
├── installer.py                # Main orchestrator
├── config/
│   └── download_urls.yaml      # Download URLs and checksums
├── utils/
│   ├── __init__.py
│   ├── progress.py             # Progress bar utilities
│   ├── network.py              # Network utilities
│   └── checksum.py             # Checksum verification
└── scripts/
    ├── build_installer_pkg.sh  # Build script
    └── install_verify.sh       # Verification script
```text

### 4. Build Process

**File**: `scripts/build_installer_pkg.sh`

```bash
#!/bin/bash

# Build the Mojo installer package
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DIST_DIR="$PROJECT_ROOT/dist"

# Create distribution directory
mkdir -p "$DIST_DIR"

# Build the package (Python or Mojo depending on implementation)
if [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
    echo "Building Python package..."
    python -m build --outdir "$DIST_DIR"
elif [ -f "$PROJECT_ROOT/mojo.yaml" ]; then
    echo "Building Mojo package..."
    mojo package "$PROJECT_ROOT/mojo_installer" -o "$DIST_DIR/mojo-installer.mojopkg"
fi

echo "Package build complete: $DIST_DIR"
ls -lh "$DIST_DIR"
```text

### 5. Integration Testing

The packaging phase includes verification that the complete installer workflow works:

```bash
#!/bin/bash
# Integration test: Complete installer workflow

# Step 1: Detect platform
echo "Detecting platform..."
platform_info=$(python -c "from mojo_installer import detect_platform; print(detect_platform())")

# Step 2: Download Mojo
echo "Downloading Mojo for $platform_info..."
python -c "from mojo_installer import download_mojo; download_mojo('$platform_info', '24.5.0')"

# Step 3: Verify download
echo "Verifying download integrity..."
if [ -f "/tmp/mojo_download/mojo-24.5.0.tar.gz" ]; then
    echo "SUCCESS: Mojo downloaded and verified"
else
    echo "FAILURE: Download did not complete"
    exit 1
fi
```text

### 6. Dependency Configuration

**Dependencies** (from test/implementation phases):

- `requests` - HTTP client for downloads (or `urllib` from stdlib)
- `tqdm` - Progress bar visualization
- `pyyaml` - Configuration file parsing
- `pytest` - Testing framework (dev only)

**Add to** `requirements.txt` or `pyproject.toml`:

```text
requests>=2.28.0
tqdm>=4.65.0
pyyaml>=6.0
```text

### 7. Production Considerations

#### Network Reliability

1. **Timeout Configuration**: Set appropriate timeouts for production networks
   - Initial connection: 10 seconds
   - Per-chunk timeout: 30 seconds
   - Total download timeout: 5 minutes (configurable)

1. **Retry Logic**:
   - Automatic retries on transient failures (3 attempts)
   - Exponential backoff between retries
   - Clear error messages for persistent failures

1. **Progress Tracking**:
   - Show download percentage
   - Show speed and estimated time
   - Show downloaded/total file size

#### Security

1. **Checksum Verification**:
   - Always verify against published checksums
   - Support multiple checksum algorithms (SHA256 primary)
   - Fail safely if checksum cannot be verified

1. **HTTPS**:
   - Always use HTTPS for downloads
   - Verify SSL certificates
   - Fail if certificate validation fails

1. **File Permissions**:
   - Set appropriate permissions on downloaded files
   - Prevent world-writable downloads
   - Remove temporary/partial downloads on failure

## Packaging Workflow

### Phase Completion Steps

1. **Verify Implementation** (#862 is complete)
   - All tests from #861 pass
   - Code follows project standards
   - Documentation is present

1. **Create Build Artifacts**
   ```bash
   # Build the package
   ./scripts/build_installer_pkg.sh

   # Verify artifacts created
   ls -lh dist/

   ```

3. **Create Verification Scripts**
   ```bash

   # Make scripts executable
   chmod +x scripts/build_installer_pkg.sh
   chmod +x scripts/install_verify.sh

   # Test installation in clean environment
   ./scripts/install_verify.sh

   ```

4. **Document Build Process**
   - Create `BUILD_PACKAGE.md` with build instructions
   - Document configuration requirements
   - Provide troubleshooting guide
   - Include verification procedures

5. **Configure Distribution**
   - Update `.gitignore` to exclude binary artifacts
   - Document package contents
   - Specify package metadata
   - Create installation instructions

### Expected Artifacts

After successful packaging:

```text

dist/
├── mojo-installer-0.1.0.tar.gz    # Source distribution
├── mojo_installer-0.1.0-py3-*.whl  # Wheel distribution (if Python)
└── mojo-installer.mojopkg          # Mojo package (if Mojo)

scripts/
├── build_installer_pkg.sh           # Build script (in git)
└── install_verify.sh                # Verification script (in git)

BUILD_PACKAGE.md                      # Documentation (in git)

```text
## Git Configuration

### .gitignore Updates

```text

# Build and distribution artifacts

dist/
build/
*.egg-info/
*.whl
*.tar.gz
*.mojopkg
__pycache__/
*.pyc

# Download cache (if implemented)

cache/
downloads/

# Temporary files

*.tmp
.tmp/

```text
### What Gets Committed

**YES - Commit These**:
- Source code (`mojo_installer/`, `*.py` files)
- Build scripts (`scripts/build_*.sh`)
- Verification scripts (`scripts/install_verify.sh`)
- Configuration templates (`config/*.yaml.example`)
- Documentation (`BUILD_PACKAGE.md`, `INSTALLATION.md`)

**NO - Do NOT Commit**:
- Binary distributions (`dist/*.tar.gz`, `dist/*.whl`, `dist/*.mojopkg`)
- Downloaded files (`/tmp/mojo_download/`)
- Build output (`build/`)
- Cache files

## Verification Checklist

Package phase completion requires:

- [ ] Binary package/distribution builds successfully
- [ ] Build script (`scripts/build_installer_pkg.sh`) created and documented
- [ ] Installation verification script (`scripts/install_verify.sh`) created
- [ ] Integration test confirms complete workflow works
- [ ] All dependencies specified in requirements/pyproject
- [ ] Configuration files include all necessary settings
- [ ] Build documentation created (`BUILD_PACKAGE.md`)
- [ ] Installation instructions documented
- [ ] `.gitignore` updated to exclude artifacts
- [ ] Package can be installed in clean environment
- [ ] All module exports accessible after installation
- [ ] End-to-end test on all supported platforms passes

## Implementation Notes

### Why This Is a Package Phase

**Correct interpretation** of packaging phase:

- Creates actual distributable artifacts (not just documentation)
- Builds packages from source code written in implementation phase
- Creates scripts for reproducible builds
- Verifies package works in clean environments
- Configures distribution and installation

**Incorrect interpretation** (what NOT to do):

- Just documenting that code exists
- Verifying source code structure
- Notes about being "ready for production" without artifacts
- Package-only documentation without build process

### Platform-Specific Considerations

The download module must handle variations across platforms:

**Linux**:
- x86_64 (Intel/AMD) - primary target
- aarch64 (ARM64) - growing importance
- Support for package managers (apt, yum, etc.)

**macOS**:
- Intel x86_64 - legacy support
- ARM64 (Apple Silicon) - primary modern target
- Code signing requirements

**Windows** (future):
- x86_64 only initially
- MSI installer vs zip distribution
- PATH configuration for Windows

## Design Decisions

### URL Strategy

**Decision**: Use configuration file for URL templates

**Rationale**:
- Enables version management without code changes
- Supports mirror/CDN failover
- Platform-specific URLs are maintainable
- Easy to test with mock URLs

### Checksum Verification

**Decision**: Always verify checksums, fail if missing

**Rationale**:
- Security requirement for binary downloads
- Detects corrupted downloads
- Prevents installation of tampered binaries
- Configuration includes checksums for all versions

### Progress Bar

**Decision**: Use `tqdm` library for progress display

**Rationale**:
- Standard library for progress bars in Python
- Works in CI/CD environments
- Shows speed and ETA
- Minimal dependency (widely available)

### Retry Strategy

**Decision**: Exponential backoff with configurable limits

**Rationale**:
- Handles transient network failures
- Doesn't overwhelm slow networks
- Respects rate limits
- User can interrupt long waits

## Testing Strategy

### Integration Tests

Complete workflow testing:

1. **Platform Detection → Download → Verification**
   - Detect actual system platform
   - Download real Mojo binary
   - Verify checksum against published values

2. **Network Simulation**
   - Mock network failures
   - Test retry logic
   - Verify error messages
   - Test timeout handling

3. **Cross-Platform Testing**
   - Test on Linux x86_64, Linux ARM64, macOS Intel, macOS ARM
   - Verify correct URLs for each platform
   - Test platform-specific handling

4. **File Integrity**
   - Test with real Mojo binaries
   - Verify checksums match published values
   - Test with corrupted files (should fail)
   - Test extraction process

## Next Steps

After completing this packaging phase:

1. **Build Package**:
   ```bash

   ./scripts/build_installer_pkg.sh

   ```

2. **Test Installation**:
   ```bash

   ./scripts/install_verify.sh

   ```

3. **Verify Integration**:
   ```bash

   # Test complete installer workflow
   python -m pytest tests/integration/test_complete_installer.py -v

   ```

4. **Create PR**:
   - Link to issue #863
   - Include build scripts and documentation
   - Do NOT commit binary artifacts
   - Reference issue in commit message

5. **Cleanup Phase** (#864):
   - Address any issues found during packaging
   - Refactor if needed
   - Finalize documentation
   - Prepare for release

## References

### Project Documentation

- [5-Phase Development Workflow](../../../../../../../notes/review/README.md)
- [Package Phase Guide](../../../../../../../agents/guides/package-phase-guide.md)
- [Project Standards](../../../../../../../CLAUDE.md)

### Mojo Documentation

- [Mojo Packaging](https://docs.modular.com/mojo/manual/packages/)
- [MAX SDK](https://docs.modular.com/max/)

### Related Issues

- **#860**: Planning and design
- **#861**: Test suite and test cases
- **#862**: Implementation code
- **#863**: Packaging (this issue)
- **#864**: Cleanup and finalization
- **#855-859**: Platform detection (dependency)

## File Locations

- **Source Plan**: `/home/mvillmow/ml-odyssey-manual/notes/plan/03-tooling/03-setup-scripts/01-mojo-installer/02-download-mojo/plan.md`
- **Issue Directory**: `/home/mvillmow/ml-odyssey-manual/notes/issues/863/`
- **Implementation Directory**: (To be determined by implementation phase)
- **Build Scripts**: `scripts/build_installer_pkg.sh`, `scripts/install_verify.sh`
- **Documentation**: `BUILD_PACKAGE.md`, `INSTALLATION.md`

---

**Created**: Documentation for GitHub Issue #863 - Mojo Download Integration and Packaging
**Phase**: Package (Phase 4 of 5-phase workflow)
**Status**: Ready for implementation
