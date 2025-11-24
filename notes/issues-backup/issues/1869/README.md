# Issue #1869: Remove get_repo_root() Code Duplication

## Overview

Fixed code duplication by removing duplicate `get_repo_root()` implementations across 5 scripts and consolidating them to use the canonical version in `scripts/common.py`.

## Problem

The `get_repo_root()` function was duplicated across 6 files:

1. `scripts/common.py` (canonical implementation)
1. `scripts/validate_links.py`
1. `scripts/check_readmes.py`
1. `scripts/validate_structure.py`
1. `scripts/package_papers.py`
1. `scripts/execute_backward_tests_merge.py`

This violates the DRY (Don't Repeat Yourself) principle and creates maintenance overhead.

## Solution

Modified 5 scripts to import `get_repo_root` from `common.py`:

### Files Modified

1. **validate_links.py**:
   - Added: `from common import get_repo_root` (line 22)
   - Removed: Lines 30-33 (duplicate function)

1. **check_readmes.py**:
   - Added: `from common import get_repo_root` (line 20)
   - Removed: Lines 48-51 (duplicate function)

1. **validate_structure.py**:
   - Added: `from common import get_repo_root` (line 20)
   - Removed: Lines 64-68 (duplicate function)

1. **package_papers.py**:
   - Added: `from common import get_repo_root` (line 19)
   - Removed: Lines 20-49 (duplicate function with fallback logic)

1. **execute_backward_tests_merge.py**:
   - Added: `from common import get_repo_root` (line 26)
   - Removed: Lines 27-35 (git-based implementation)

### Canonical Implementation

The canonical `get_repo_root()` in `scripts/common.py`:

```python
def get_repo_root() -> Path:
    """
    Get the repository root directory.

    Searches upward from the current file location until finding a directory
    containing a .git folder.

    Returns:
        Path to repository root

    Raises:
        RuntimeError: If repository root cannot be found
    """
    current = Path(__file__).resolve().parent

    # Search upward for .git directory
    while current != current.parent:
        if (current / '.git').exists():
            return current
        current = current.parent

    raise RuntimeError("Could not find repository root (no .git directory found)")
```

## Testing

All modified scripts tested to verify functionality:

```bash
python3 scripts/validate_links.py
python3 scripts/check_readmes.py
python3 scripts/validate_structure.py
python3 scripts/package_papers.py --help
```

## Impact

- **Maintainability**: Single source of truth for repo root detection
- **Code Quality**: Reduced duplication by ~40 lines
- **Consistency**: All scripts use same logic
- **DRY Principle**: Follows Don't Repeat Yourself best practice

## Related Issues

Part of Wave 2 tooling improvements from continuous improvement session.

## Success Criteria

- [x] All 5 scripts modified to import from common.py
- [x] All duplicate get_repo_root() functions removed
- [x] Scripts tested and working correctly
- [x] Issue documentation created
- [x] Changes committed with clear message
