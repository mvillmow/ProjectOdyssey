# Issue #858: [Package] Detect Platform - Integration and Packaging

## Objective

Integrate the platform detection implementation with the Mojo installer codebase and ensure proper packaging for
deployment. This packaging phase verifies compatibility with downstream components, configures dependencies, and
prepares the platform detection module for distribution in the ML Odyssey tooling suite.

## Deliverables

- Integrated platform detection module in Mojo installer package
- Configuration and metadata for module dependencies
- Integration tests validating platform detection in complete installer workflow
- Package manifest and distribution configuration
- Installation verification scripts and documentation
- CI/CD integration for packaging validation

## Success Criteria

- [ ] Platform detection module correctly identifies major platforms (Linux x86_64, Linux arm64, macOS Intel, macOS ARM)
- [ ] Architecture detection works accurately on all target systems
- [ ] Warning and error messages display properly for unsupported platforms
- [ ] Platform identifiers are consistent and normalized across all integration points
- [ ] Module integrates seamlessly with Mojo download component (issues #860-863)
- [ ] All Python dependencies are properly configured (stdlib only, documented version requirements)
- [ ] Package installs correctly in clean environments using standard Python tooling
- [ ] Integration tests pass on all supported platforms
- [ ] Package metadata is complete (setup.py, pyproject.toml, or equivalent)

## References

- **Source Plan**: `/notes/plan/03-tooling/03-setup-scripts/01-mojo-installer/01-detect-platform/plan.md`
- **Planning Phase**: Issue #855 [Plan] Detect Platform - Design and Documentation
- **Test Phase**: Issue #856 [Test] Detect Platform - Write Tests
- **Implementation Phase**: Issue #857 [Impl] Detect Platform - Implementation
- **Cleanup Phase**: Issue #859 [Cleanup] Detect Platform - Refactor and Finalize
- **Downstream Components**: Issues #860-863 (Mojo Download and installation components)
- **Workflow Documentation**: `/CLAUDE.md#5-phase-development-workflow`
- **Python Standards**: `/CLAUDE.md#python-coding-standards`

## Implementation Notes

This packaging phase focuses on integrating the platform detection module with the larger Mojo installer
architecture and ensuring it can be properly deployed and distributed.

### 1. Integration Architecture

The platform detection module integrates into the broader Mojo installer ecosystem:

```text
mojo_installer/
├── __init__.py                          # Package initialization
├── platform_detector.py                 # This module (Issue #857)
├── download_manager.py                  # Consumes platform info (Issues #860-863)
├── installer.py                         # Main orchestration script
├── config.py                            # Configuration management
└── utils/
    ├── logging.py                       # Logging utilities
    └── errors.py                        # Custom exceptions
```text

### Integration Points

1. **Installer Main Script** (`installer.py`)
   - Import and call `platform_detector.detect_platform()`
   - Handle platform detection errors gracefully
   - Pass platform details to download manager
   - Log platform information for debugging

1. **Download Manager** (Issues #860-863)
   - Receives platform identifier from platform detection
   - Uses architecture to determine correct binary URL
   - Validates supported platform/architecture combination
   - Falls back to user input if detection fails

1. **Configuration Management** (`config.py`)
   - Store detected platform in installation config
   - Cache platform info to avoid repeated detection
   - Allow user override if needed

### 2. Integration Steps

The platform detection module must complete the following operations:

#### Step 1: Query Operating System Information

Using Python's standard `platform` module:

```python
import platform

system = platform.system()      # Returns: 'Linux', 'Darwin', 'Windows', etc.
version = platform.version()    # Full version string
release = platform.release()    # Release information
```text

### Expected Outputs

- Linux → "linux"
- Darwin → "macos"
- Windows → "windows" (unsupported, should warn)
- Other → handled as unsupported

#### Step 2: Detect Processor Architecture

Using `platform.machine()`:

```python
machine = platform.machine()    # Returns: 'x86_64', 'aarch64', 'arm64', etc.
```text

### Expected Outputs

- x86_64, AMD64 → "x86_64"
- aarch64, arm64 → "arm64"
- Other → handled as unsupported

#### Step 3: Check OS Version Compatibility

Version checking strategy:

```python
# Linux version info
if system == 'Linux':
    # Use platform.version() or os.uname()
    # Check minimum kernel version or glibc version
    # Log warnings for very old versions

# macOS version info
if system == 'Darwin':
    # Use platform.mac_ver() for detailed macOS version
    # Check minimum macOS version (e.g., 10.13+)
    # Handle both Intel and Apple Silicon versions
```text

#### Step 4: Return Structured Platform Details

The module returns a consistent data structure:

```python
{
    "platform": "linux" | "macos",      # Lowercase, normalized
    "architecture": "x86_64" | "arm64",
    "os_version": "string with details",
    "compatible": True | False,         # Overall compatibility assessment
    "warnings": ["list", "of", "warnings"],  # Any compatibility concerns
    "supported": True | False           # Is this officially supported?
}
```text

### 3. Dependency Configuration

The platform detection module has **MINIMAL DEPENDENCIES**:

```text
Runtime Dependencies:
- Python >=3.7          # Core requirement
- stdlib only           # No external packages

Development Dependencies (for testing/packaging):
- pytest >=6.0          # Test execution
- pytest-cov            # Coverage reporting
- build                 # Package building
- twine                 # Package distribution
```text

### Rationale

- Using only stdlib ensures the module can run in any Python environment
- No version conflicts with other components
- Reduces installation burden on users
- Platform module has been stable since Python 2.6

**Package Metadata** (`setup.py` or `pyproject.toml`):

```toml
[project]
name = "mojo-installer-platform"
version = "0.1.0"
description = "Platform detection for Mojo installer"
requires-python = ">=3.7"
dependencies = []  # No external dependencies!
```text

### 4. Package Structure and Distribution

#### Directory Layout

```text
mojo-installer/
├── src/
│   └── mojo_installer/
│       ├── __init__.py
│       ├── platform_detector.py    # Core module (from Issue #857)
│       ├── download_manager.py
│       └── installer.py
├── tests/
│   ├── test_platform_detection.py   # From Issue #856
│   ├── test_architecture_detection.py
│   ├── test_version_detection.py
│   └── test_error_handling.py
├── setup.py
├── pyproject.toml
├── README.md
└── LICENSE
```text

#### Package Manifest

The package manifest (`MANIFEST.in` or `pyproject.toml`) should include:

```toml
[tool.setuptools]
packages = ["mojo_installer"]

[tool.setuptools.package-data]
mojo_installer = ["py.typed"]  # Include type hints

[project.scripts]
mojo-install = "mojo_installer.installer:main"
```text

#### Distribution Configuration

### Python Packaging Standards

1. **Setup file** (`setup.py`):
   - Define package metadata
   - Specify dependencies
   - Configure entry points
   - Allow installation via `pip install`

1. **Configuration file** (`pyproject.toml`):
   - Modern Python packaging standard
   - Declares build requirements
   - Includes project metadata
   - Enables isolation during builds

1. **Distribution artifacts**:
   - Source distribution (`.tar.gz`)
   - Wheel distribution (`.whl`)
   - Upload to PyPI for distribution

### 5. Integration Testing Strategy

Integration tests validate the platform detection module within the complete installer workflow:

#### Test Categories

1. **Platform Detection Tests** (from Issue #856)
   - All major platforms tested
   - Mock data for architecture variations
   - Unsupported platform handling

1. **Integration Tests** (new for packaging phase)
   - Complete installer workflow
   - Platform info passed correctly to download manager
   - Download URLs constructed correctly based on detected platform
   - User experience on each platform type

1. **Cross-Platform Tests**
   - Linux x86_64 detection
   - Linux arm64 detection (test on hardware or skip gracefully)
   - macOS Intel detection
   - macOS ARM detection (test on hardware or skip gracefully)
   - Windows detection with graceful unsupported message

1. **End-to-End Tests**
   - Full installation workflow simulation
   - Verify correct binaries would be downloaded
   - Test on all CI/CD platforms (Ubuntu, macOS runners)

#### Test Execution Matrix

```text
Platform      | Architecture | Test Strategy
--- | --- | ---
Linux         | x86_64       | Run in CI/CD (GitHub Actions Ubuntu runner)
Linux         | arm64        | Skip or use QEMU emulation
macOS         | Intel        | Run in CI/CD (GitHub Actions macOS runner)
macOS         | Apple Silicon| Run in CI/CD (GitHub Actions macOS ARM runner) or skip
Windows       | x86_64       | Run in CI/CD (GitHub Actions Windows runner)
```text

### 6. Distribution and Installation Verification

#### Installation Methods

The packaged module should support installation via:

```bash
# From PyPI (future)
pip install mojo-installer

# From source
pip install ./mojo-installer

# In development mode
pip install -e ./mojo-installer

# With extras for development
pip install -e ./mojo-installer[dev]
```text

#### Verification Scripts

Create scripts to verify successful installation:

```bash
# Verify platform detection works
python -c "from mojo_installer.platform_detector import detect_platform; print(detect_platform())"

# Verify integration with download manager
python -c "from mojo_installer import installer; print(installer.validate_environment())"

# Run full integration test
pytest tests/test_integration.py -v
```text

### 7. CI/CD Integration

The packaging phase should integrate with CI/CD pipelines:

```yaml
# GitHub Actions workflow (.github/workflows/package-detect-platform.yml)

name: Package Platform Detection

on: [push, pull_request]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, macos-14]  # Latest macOS + ARM
        python-version: ['3.7', '3.8', '3.9', '3.10', '3.11']

    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Build distribution
        run: python -m build

      - name: Test installation
        run: pip install dist/*.whl

      - name: Verify platform detection
        run: python -m pytest tests/test_integration.py -v

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: dist-${{ matrix.os }}-py${{ matrix.python-version }}
          path: dist/
```text

### 8. Package Metadata and Documentation

#### README for Package

Create `mojo_installer/README.md`:

```markdown
# Mojo Installer - Platform Detection Module

## Purpose

Detect the current operating system, architecture, and version information to
determine which Mojo binaries to download and install.

## Installation

```bash

pip install mojo-installer

```text
## Usage

```python

from mojo_installer.platform_detector import detect_platform

platform_info = detect_platform()
print(f"Platform: {platform_info['platform']}")
print(f"Architecture: {platform_info['architecture']}")

```text
## Supported Platforms

- Linux (x86_64, ARM64)
- macOS (Intel, Apple Silicon)

## Unsupported Platforms

- Windows (not yet supported)
- Other architectures

## Error Handling

The module provides clear error messages for unsupported platforms:

```python

try:
    info = detect_platform()
except UnsupportedPlatformError as e:
    print(f"Installation not supported: {e}")

```text
## Contributing

See main project README for contribution guidelines.
```text

#### Version and Release Notes

Document the version:

```python
# mojo_installer/__init__.py
__version__ = "0.1.0"
__author__ = "ML Odyssey Contributors"
__license__ = "MIT"
```text

### 9. Common Packaging Issues and Solutions

#### Issue: Platform Detection Works Locally but Fails in CI

**Solution**: Ensure tests mock the platform.system() and platform.machine() calls to test
different platform combinations in all environments.

#### Issue: Package Installation Fails Due to Missing Dependencies

**Solution**: Keep dependencies minimal (stdlib only). Verify setup.py/pyproject.toml correctly
declares zero external dependencies.

#### Issue: Architecture Detection Returns Unexpected Values

**Solution**: Create a mapping of known aliases (aarch64 → arm64, AMD64 → x86_64) in the module.

#### Issue: OS Version Detection Fails

**Solution**: Implement graceful fallback - if version detection fails, still return platform and
architecture but mark version as "unknown" and compatibility as "uncertain".

### 10. Packaging Checklist

Before marking this issue complete:

- [ ] Platform detection module imported successfully in installer
- [ ] All tests from Issue #856 pass
- [ ] Integration tests created and passing
- [ ] Package metadata configured (setup.py/pyproject.toml)
- [ ] Distribution artifacts build successfully (wheel, sdist)
- [ ] Installation verification scripts work on all platforms
- [ ] Documentation complete and accurate
- [ ] CI/CD pipeline configured for packaging validation
- [ ] Cross-platform testing completed (or CI proves compatibility)
- [ ] No external dependencies required (stdlib only)
- [ ] Platform identifiers normalized and consistent
- [ ] Error messages clear and actionable
- [ ] Version information properly configured

## Implementation Notes

As implementation proceeds, findings and decisions will be documented here:

### Platform Detection Findings

*To be filled during implementation*

### Integration Challenges

*To be filled during implementation*

### Performance Observations

*To be filled during implementation*

### Testing Insights

*To be filled during implementation*

### Distribution Lessons

*To be filled during implementation*

## Next Phase: Cleanup (Issue #859)

After packaging is complete, the cleanup phase will:

1. Refactor for code quality and consistency
1. Optimize for performance
1. Enhance error messages based on real-world usage
1. Improve test coverage to 95%+
1. Create final documentation and usage guides
1. Validate cross-platform deployment

See Issue #859 for detailed cleanup requirements.
