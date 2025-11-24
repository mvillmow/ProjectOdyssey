# Issue #857: [Impl] Detect Platform - Implementation

## Objective

Implement the platform detection functionality to accurately identify operating system, architecture, and
version details. Build clean, maintainable code that passes all tests and follows Python best practices for
system detection. This is the implementation phase of the platform detection component, which is critical
for the Mojo installer to download and install the correct binary.

## Deliverables

- Platform detection module implementation (`platform_detector.py`)
- OS identification functions
- Architecture detection functions
- Version compatibility checking
- Error handling for unsupported platforms
- Comprehensive inline documentation and docstrings
- Module integration and API export

## Success Criteria

- [ ] Implementation correctly identifies major platforms (Linux, macOS)
- [ ] Architecture detection is accurate (x86_64, arm64)
- [ ] Warnings are displayed for unsupported platforms
- [ ] Platform identifiers are consistent and standardized
- [ ] All tests pass (from issue #856)
- [ ] Code follows Python best practices and style guidelines
- [ ] Comprehensive docstrings and type hints are included
- [ ] Error messages are clear and actionable
- [ ] No external dependencies beyond Python stdlib

## References

- **Parent component**: Detect Platform (notes/plan/03-tooling/03-setup-scripts/01-mojo-installer/01-detect-platform/)
- **Related workflow issues**:
  - #855 [Plan] Detect Platform - Design and Documentation
  - #856 [Test] Detect Platform - Write Tests
  - #858 [Package] Detect Platform - Integration/Packaging
  - #859 [Cleanup] Detect Platform - Finalization
- **Language justification**: ADR-001 Pragmatic Hybrid Language Approach (Python for tooling/automation with subprocess needs)
- **Python coding standards**: CLAUDE.md Python Coding Standards section
- **Plan document**: notes/plan/03-tooling/03-setup-scripts/01-mojo-installer/01-detect-platform/plan.md

## Implementation Notes

This implementation should include:

### 1. Module Structure

Create `platform_detector.py` in the appropriate location within the installer module:

```text
scripts/install/
├── __init__.py
├── platform_detector.py    # This implementation
├── version_detector.py      # Related module
└── installer.py             # Main installer
```text

### 2. Core Functions

**Primary Function**: `detect_platform()`

- Input: None (system detection)
- Output: Structured dictionary with platform information
- Should be the primary public API

### Helper Functions

- `_get_os_type()` - Query OS using `platform.system()`
- `_normalize_platform_name()` - Standardize platform identifier
- `_detect_architecture()` - Query architecture using `platform.machine()`
- `_normalize_architecture()` - Standardize architecture naming
- `_get_os_version()` - Extract version information
- `_assess_compatibility()` - Check if platform is supported

### 3. Data Structure (Output Format)

All platform detection should return this standardized structure:

```python
{
    "platform": "linux" | "macos",      # Lowercase, standardized
    "architecture": "x86_64" | "arm64",  # Normalized naming
    "os_version": "20.04" | "14.1",      # Version string
    "compatible": True | False,          # Is this a supported platform?
    "warnings": ["message1", "message2"], # Any compatibility warnings
    "details": {                         # Additional metadata
        "platform_raw": "Linux",         # Raw OS name from system
        "arch_raw": "x86_64",            # Raw architecture from system
    }
}
```text

### 4. Supported Platforms Matrix

### Linux Support

- Architectures: x86_64 (AMD64), ARM64 (aarch64)
- Distributions: Ubuntu, Debian, RHEL/CentOS, Fedora (via generic Linux detection)
- Version tracking: Capture distribution version when available

**macOS Support**:

- Architectures: x86_64 (Intel), ARM64 (Apple Silicon)
- Version tracking: Capture macOS version (e.g., "14.1" for Sonoma)

### Unsupported Platforms

- Windows - Should warn and set `compatible: False`
- Other architectures (i386, ppc64, etc.) - Should warn and set `compatible: False`

### 5. Error Handling Strategy

```python
# Clear exceptions for critical failures
class UnsupportedPlatformError(Exception):
    """Raised when platform is not supported."""
    pass

class PlatformDetectionError(Exception):
    """Raised when platform detection fails."""
    pass
```text

### Error Message Examples

- `"Unsupported platform 'Windows'. Please use Linux or macOS."`
- `"Unsupported architecture 'i386'. Please use x86_64 or arm64."`
- `"Unable to detect OS version. Some features may be unavailable."`

### 6. Code Quality Standards

### Type Hints

```python
def detect_platform() -> Dict[str, Any]:
    """Detect system platform information."""
    pass

def _get_os_type() -> str:
    """Get normalized OS type."""
    pass
```text

**Docstrings** (Google style):

```python
def detect_platform() -> Dict[str, Any]:
    """Detect and return system platform information.

    This function queries the system to determine the operating system,
    processor architecture, and version details. It returns a standardized
    dictionary that can be used by the installer to select appropriate
    binaries.

    Returns:
        Dict[str, Any]: Platform information with keys:
            - platform: OS identifier (linux, macos)
            - architecture: Architecture identifier (x86_64, arm64)
            - os_version: Version string
            - compatible: Boolean indicating support status
            - warnings: List of compatibility warnings
            - details: Additional metadata

    Raises:
        PlatformDetectionError: If platform detection fails
    """
```text

### Style Guidelines

- PEP 8 compliance (enforced by pre-commit hooks)
- Maximum line length: 120 characters
- Use logging module for warnings (not print)
- Import organization: stdlib, then third-party, then local

### 7. Testing Integration Points

This implementation should work seamlessly with tests from #856:

- Unit tests for each helper function
- Mocked `platform` module calls for different platforms
- Integration tests for complete detection workflow
- Edge case handling (unknown platforms, missing info)

Test files to integrate with:

- `tests/installer/test_platform_detection.py`
- `tests/installer/test_architecture_detection.py`
- `tests/installer/test_error_handling.py`

### 8. Implementation Approach (TDD-First)

1. **Start with test review** - Understand what #856 expects
1. **Implement incrementally**:
   - Step 1: Implement `_get_os_type()`
   - Step 2: Implement `_detect_architecture()`
   - Step 3: Implement `_get_os_version()`
   - Step 4: Implement `_assess_compatibility()`
   - Step 5: Integrate into `detect_platform()`
1. **Run tests after each step** - Ensure TDD compliance
1. **Cross-platform validation** - Test on Linux (primary), verify macOS compatibility

### 9. Logging and Debugging

Use Python's logging module for diagnostics:

```python
import logging

logger = logging.getLogger(__name__)

# In functions
logger.debug(f"Detected platform: {platform}")
logger.warning(f"Platform may not be fully supported: {platform}")
logger.error(f"Failed to detect architecture: {e}")
```text

### 10. No External Dependencies

This module MUST only use Python stdlib:

- `platform` - OS/architecture detection
- `logging` - Diagnostics
- `typing` - Type hints
- Standard data structures (dict, list, etc.)

Do NOT add pip dependencies unless absolutely necessary (and document justification).

## Implementation Workflow

1. **Review tests** from #856 to understand expected behavior
1. **Create module file** with stub functions
1. **Implement OS detection** - Use platform.system()
1. **Implement architecture detection** - Use platform.machine()
1. **Implement version detection** - Parse version strings
1. **Implement compatibility assessment** - Check support matrix
1. **Add error handling** - Clear exceptions and messages
1. **Add logging** - Debug-level diagnostics
1. **Run tests** - Validate all #856 tests pass
1. **Code review** - Ensure PEP 8 compliance and documentation quality
