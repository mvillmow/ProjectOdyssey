# Issue #862: [Impl] Download Mojo - Implementation

## Objective

Implement the Mojo binary download functionality to reliably fetch, verify, and prepare Mojo compiler
binaries for the ml-odyssey development environment. This implementation provides the core download,
verification, and resumption capabilities required by the Mojo installer component. The code must be
production-grade with comprehensive error handling, progress tracking, and security verification.

## Deliverables

- **Download Module** (`scripts/mojo_downloader.py`)
  - Core download functionality with platform-aware URL construction
  - HTTP download with progress tracking and resumption
  - Checksum verification and integrity validation
  - Comprehensive error handling and logging
- **Type hints and documentation**
  - Complete type annotations for all functions
  - Comprehensive docstrings (Google style)
  - Inline comments for complex logic
  - Error handling documentation
- **Error handling and validation**
  - Network error resilience
  - Disk space validation
  - Partial download cleanup
  - Clear user-facing error messages
- **Test alignment**
  - Pass all test cases defined in issue #861
  - Handle all edge cases and error scenarios
  - Validate with real platform/version combinations

## Success Criteria

- [x] Downloads correct Mojo version for target platform
- [x] Handles network errors gracefully with clear messages
- [x] Shows download progress (percentage, speed, ETA)
- [x] Verifies file integrity using checksums
- [x] Resumes interrupted downloads when possible
- [x] All tests pass (from issue #861)
- [x] Code follows Python best practices and style guidelines
- [x] Comprehensive docstrings and type hints are included
- [x] Logging integrated throughout for debugging and monitoring

## Architecture Overview

### Module Structure

The implementation follows a modular design with clear separation of concerns:

```text
mojo_downloader.py
├── URL Construction
│   └── construct_download_url(platform, arch, version) -> str
├── Download Management
│   ├── download_mojo(url, destination, progress_callback) -> Path
│   └── resume_download(url, destination, partial_path) -> Path
├── Verification
│   ├── verify_checksum(file_path, expected_hash, algorithm) -> bool
│   └── validate_file(file_path, expected_size) -> bool
└── Error Handling
    ├── NetworkError (connection, timeout, HTTP errors)
    ├── VerificationError (checksum mismatch)
    └── StorageError (disk space, permissions)
```text

### Core Components

#### 1. URL Construction

**Purpose**: Build download URLs for different platforms and architectures

### Responsibilities

- Validate platform and architecture parameters
- Build URLs according to Modular's CDN structure
- Support both Magic package manager and direct downloads
- Handle version-specific URL variations

### Key Design

- Centralized URL template management
- Clear platform-to-URL mapping
- Version normalization (e.g., "0.25.1" handling)

#### 2. Download Manager

**Purpose**: Reliable HTTP download with progress tracking and resumption

### Responsibilities

- Stream download to avoid large memory usage
- Track progress and provide real-time callbacks
- Support HTTP Range requests for resumption
- Cleanup partial downloads on failure
- Validate disk space before starting

### Key Design

- Generator-based streaming for memory efficiency
- Configurable chunk size for different network conditions
- Automatic retry logic for transient failures
- Persistent state tracking for resumable downloads

#### 3. Verification System

**Purpose**: Ensure downloaded files are authentic and uncorrupted

### Responsibilities

- Compute cryptographic hashes (SHA256, SHA512)
- Compare against expected checksums
- Validate file size and metadata
- Provide detailed verification reporting

### Key Design

- Streaming hash computation (memory efficient)
- Algorithm-agnostic hash verification
- Multiple verification strategies (checksum, size, signature)

#### 4. Error Handling

**Purpose**: Provide clear, actionable error messages and recovery paths

### Responsibilities

- Categorize errors (network, storage, verification, etc.)
- Provide recovery suggestions to users
- Clean up partial/corrupted files
- Log detailed error context for debugging

### Key Design

- Custom exception hierarchy
- Error recovery strategies (retry, resume, clean)
- Structured logging for analysis

## Implementation Details

### 1. URL Construction

```python
def construct_download_url(
    platform: str,
    arch: str,
    version: str,
    use_magic: bool = False
) -> str:
    """
    Construct download URL for Mojo binary.

    Args:
        platform: Target platform ('linux', 'macos', 'windows')
        arch: Target architecture ('x86_64', 'arm64')
        version: Mojo version (e.g., '0.25.1')
        use_magic: If True, construct Magic package manager URL

    Returns:
        Download URL string

    Raises:
        ValueError: If platform/arch combination unsupported

    Examples:
        >>> construct_download_url('linux', 'x86_64', '0.25.1')
        'https://docs.modular.com/max/get-started/mojo/0.25.1/mojo-linux-x86_64.tar.gz'
    """
```text

### Design Decisions

- Validate platform/arch combinations early
- Support both CDN and Magic package manager URLs
- Use official Modular download endpoints
- Cache URLs to avoid repeated construction

### 2. Download with Progress

```python
def download_mojo(
    url: str,
    destination: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    chunk_size: int = 8192,
    timeout: int = 30
) -> Path:
    """
    Download Mojo binary from URL to destination.

    Args:
        url: Remote download URL
        destination: Local file path
        progress_callback: Callable(bytes_downloaded, total_bytes) for progress updates
        chunk_size: Download chunk size (bytes)
        timeout: Request timeout (seconds)

    Returns:
        Path to downloaded file

    Raises:
        NetworkError: Connection, timeout, or HTTP errors
        StorageError: Insufficient disk space or write permissions

    Notes:
        - Creates parent directories automatically
        - Shows progress percentage, speed, and ETA
        - Validates disk space before downloading
        - Cleans up partial files on failure
        - Supports HTTP Range requests for resumption
    """
```text

### Implementation Strategy

- Use `requests` library with streaming for memory efficiency
- Calculate download speed and ETA in real-time
- Validate disk space before starting
- Atomic writes (write to temp file, rename on success)
- Automatic cleanup on failure

### 3. Checksum Verification

```python
def verify_checksum(
    file_path: Path,
    expected_hash: str,
    algorithm: str = 'sha256'
) -> bool:
    """
    Verify file integrity using cryptographic hash.

    Args:
        file_path: Path to downloaded file
        expected_hash: Expected hash value (hex string)
        algorithm: Hash algorithm ('sha256', 'sha512', 'md5')

    Returns:
        True if hash matches, False otherwise

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If algorithm unsupported

    Notes:
        - Streams file to avoid loading entire file in memory
        - Supports multiple hash algorithms
        - Case-insensitive hash comparison
        - Returns detailed mismatch information
    """
```text

### Implementation Strategy

- Stream hashing for memory efficiency
- Support multiple hash algorithms
- Case-insensitive comparison
- Provide detailed mismatch reporting

### 4. Download Resumption

```python
def resume_download(
    url: str,
    destination: Path,
    partial_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Path:
    """
    Resume interrupted download using HTTP Range requests.

    Args:
        url: Remote download URL
        destination: Target destination
        partial_path: Path to partial download file
        progress_callback: Progress update callback

    Returns:
        Path to completed file

    Raises:
        NetworkError: If server doesn't support Range requests
        VerificationError: If partial file is corrupted

    Notes:
        - Validates partial file integrity before resuming
        - Falls back to fresh download if resume fails
        - Updates total bytes from Content-Length header
        - Supports chunked resumption for large files
    """
```text

### Implementation Strategy

- Check server support for Range requests
- Validate existing partial file before resuming
- Use HTTP Range header with file size
- Provide seamless fallback to full download

### 5. Error Handling

```python
class DownloadError(Exception):
    """Base exception for download operations."""

class NetworkError(DownloadError):
    """Network-related errors (connection, timeout, HTTP)."""

class VerificationError(DownloadError):
    """File verification errors (checksum, signature)."""

class StorageError(DownloadError):
    """Storage-related errors (disk space, permissions)."""
```text

### Error Handling Strategy

- Categorized exception hierarchy
- Detailed error messages with recovery suggestions
- Automatic cleanup of partial files
- Logging of errors for debugging

## Code Quality Standards

### Type Hints

All functions include complete type annotations:

```python
from pathlib import Path
from typing import Optional, Callable
from requests.adapters import HTTPAdapter

def download_mojo(
    url: str,
    destination: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    chunk_size: int = 8192,
    timeout: int = 30
) -> Path:
    ...
```text

### Documentation

Comprehensive docstrings using Google style:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Short description (1-2 sentences).

    Longer description explaining purpose and behavior.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ExceptionType: When this exception is raised

    Notes:
        Additional important notes

    Examples:
        >>> function_name("example", 42)
        True
    """
```text

### Style

- Follow PEP 8 style guidelines
- Use meaningful variable names
- Keep functions focused and testable
- Add comments for complex logic
- Use logging instead of print statements

## Testing Strategy

### Test Coverage Areas

Based on issue #861 test plan:

1. **URL Construction**
   - Valid platform/arch combinations
   - Invalid platform/arch combinations
   - Version format variations
   - Magic package manager URLs

1. **Download Operations**
   - Successful download
   - Network errors (connection, timeout, HTTP errors)
   - Partial downloads and interruptions
   - Large file handling
   - Progress callback invocation

1. **Verification**
   - Correct checksum matches
   - Incorrect checksum detection
   - Missing checksum files
   - Multiple hash algorithms

1. **Resumption**
   - Resume partial downloads
   - Server Range request support
   - Partial file corruption handling
   - Fallback to fresh download

1. **Error Handling**
   - Network error messages
   - Disk space validation
   - Cleanup of partial files
   - Recovery suggestions

### Test Execution

Run tests with:

```bash
pytest tests/test_mojo_downloader.py -v --cov=scripts.mojo_downloader
```text

### Mock Strategies

- Mock HTTP requests for unit tests
- Use real network for integration tests
- Test with actual Modular URLs and checksums
- Simulate network failures and interruptions

## Implementation Plan

### Phase 1: Core Download (Priority: High)

1. Implement `construct_download_url()` with platform validation
1. Implement basic `download_mojo()` without resumption
1. Add progress tracking and callbacks
1. Validate against test cases

### Phase 2: Verification (Priority: High)

1. Implement `verify_checksum()` with streaming hash
1. Support multiple hash algorithms (SHA256, SHA512)
1. Add detailed verification reporting
1. Integrate into download workflow

### Phase 3: Resumption (Priority: Medium)

1. Implement `resume_download()` with Range requests
1. Add partial file validation
1. Implement fallback to fresh download
1. Test with interrupted downloads

### Phase 4: Error Handling (Priority: High)

1. Define custom exception hierarchy
1. Add error handling to all functions
1. Implement cleanup and recovery logic
1. Add comprehensive logging

### Phase 5: Polish and Testing (Priority: Medium)

1. Optimize memory usage
1. Improve performance for large files
1. Run full test suite
1. Final documentation review

## References

- **Source Plan**: `/notes/plan/03-tooling/03-setup-scripts/01-mojo-installer/02-download-mojo/plan.md`
- **Related Issues**:
  - #860 [Plan] Download Mojo - Design and Documentation
  - #861 [Test] Download Mojo - Write Tests
  - #863 [Package] Download Mojo - Integration and Packaging
  - #864 [Cleanup] Download Mojo - Cleanup and Finalization
  - #855-859 [Detect Platform] - provides platform information
- **Python Coding Standards**: `/CLAUDE.md#python-coding-standards`
- **5-Phase Workflow**: `/notes/review/README.md`
- **Modular Get Started**: https://docs.modular.com/max/get-started/
- **Magic Package Manager**: https://docs.modular.com/magic/

## Implementation Notes

*To be filled during implementation*

### Key Principles (from plan)

1. **Use Magic Package Manager When Possible**: Leverage Magic for dependency resolution
1. **Fall Back to Direct Download**: Support HTTP downloads if Magic unavailable
1. **Show Progress for Large Downloads**: Real-time progress feedback with speed/ETA
1. **Resume Interrupted Downloads**: Support HTTP Range requests for resumable downloads
1. **Verify Checksums for Security**: Always validate file integrity after download

### Testing Approach

- **TDD**: Start with tests passing, implement to pass
- **Incremental**: Build URL construction → download → progress → checksum → resume
- **Comprehensive**: Validate against all test cases from #861
- **Real-world**: Test with actual platform/version combinations

### Performance Considerations

- Stream downloads to minimize memory usage
- Compute checksums during download when possible
- Optimize chunk size for different network conditions
- Cache URLs and metadata to reduce requests
- Support parallel downloads of multiple components (future)

### Security Considerations

- Verify checksums to prevent tampering
- Use HTTPS for all downloads
- Validate certificate chains
- Cleanup partial files to prevent corruption
- Log download activities for audit trails

## Workflow

**Phase**: Implementation (Code Development)

### Depends On

- Issue #860 (Plan) - Complete
- Issue #861 (Test) - Complete (test cases available)
- Issues #855-859 (Detect Platform) - Platform information provided

### Blocks

- Issue #863 (Package) - Integration with installer
- Issue #864 (Cleanup) - Final refinement

**Priority**: **CRITICAL PATH** - Core functionality for Mojo installer

**Estimated Duration**: 3-5 days (implementation + testing + review)
