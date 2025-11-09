#!/usr/bin/env python3
"""
Create papers/ directory at repository root.

This script creates the papers/ directory which will contain all individual
paper implementations, with each paper in its own subdirectory.

Usage:
    python scripts/foundation/create_papers_dir.py
"""

import sys
from pathlib import Path
from typing import Optional


def create_papers_directory(repo_root: Path) -> Path:
    """
    Create the papers/ directory at the repository root.

    This function creates the papers/ directory using mkdir with parents=True
    and exist_ok=True to ensure idempotent behavior. The directory will have
    standard permissions allowing read, write, and execute operations.

    Args:
        repo_root: Path to the repository root directory

    Returns:
        Path to the created papers/ directory

    Raises:
        PermissionError: If the user lacks permissions to create the directory
        OSError: If directory creation fails for other reasons

    Examples:
        >>> from pathlib import Path
        >>> repo_root = Path("/home/user/ml-odyssey")
        >>> papers_dir = create_papers_directory(repo_root)
        >>> print(papers_dir)
        /home/user/ml-odyssey/papers
    """
    # Validate input
    if not repo_root.exists():
        raise FileNotFoundError(f"Repository root does not exist: {repo_root}")

    if not repo_root.is_dir():
        raise NotADirectoryError(f"Repository root is not a directory: {repo_root}")

    # Create papers directory
    papers_dir = repo_root / "papers"

    try:
        # parents=True: Create parent directories if needed
        # exist_ok=True: Don't raise error if directory already exists (idempotent)
        papers_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied: cannot create directory {papers_dir}"
        ) from e
    except OSError as e:
        raise OSError(f"Failed to create directory {papers_dir}: {e}") from e

    return papers_dir


def get_repo_root() -> Path:
    """
    Determine the repository root directory.

    This function finds the repository root by looking for the .git directory
    or by using the project structure conventions.

    Returns:
        Path to the repository root directory

    Raises:
        RuntimeError: If repository root cannot be determined
    """
    # Start from script location and walk up to find repo root
    current = Path(__file__).resolve()

    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        # Check for .git directory
        if (parent / ".git").exists():
            return parent
        # Check for characteristic files/directories
        if (parent / "scripts").exists() and (parent / "pyproject.toml").exists():
            return parent

    # Fallback: assume script is in scripts/foundation/
    # So repo root is two levels up
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "foundation" and script_dir.parent.name == "scripts":
        return script_dir.parent.parent

    raise RuntimeError(
        "Could not determine repository root. "
        "Please run this script from within the repository."
    )


def main() -> int:
    """
    Main entry point for the script.

    Creates the papers/ directory at the repository root and reports the result.

    Returns:
        0 on success, 1 on failure
    """
    try:
        # Determine repository root
        repo_root = get_repo_root()
        print(f"Repository root: {repo_root}")

        # Create papers directory
        papers_dir = create_papers_directory(repo_root)

        # Report success
        print(f"Successfully created papers/ directory at: {papers_dir}")

        # Verify the directory exists
        if papers_dir.exists() and papers_dir.is_dir():
            print("Verification: papers/ directory exists and is accessible")
            return 0
        else:
            print("ERROR: Directory creation reported success but verification failed")
            return 1

    except PermissionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Please check your permissions and try again.", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
