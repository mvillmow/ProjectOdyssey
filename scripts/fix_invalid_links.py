#!/usr/bin/env python3
"""
Fix invalid absolute path links in markdown files.

This script fixes two types of invalid links:
1. Full system paths: /home/mvillmow/ml-odyssey-manual/... -> relative paths
2. Absolute paths starting with /: /agents/... -> agents/...

Usage:
    python3 scripts/fix_invalid_links.py [--dry-run]
"""

import re
import sys
from pathlib import Path
from typing import Tuple

def fix_system_path_links(content: str) -> Tuple[str, int]:
    """
    Fix links with full system paths like /home/mvillmow/ml-odyssey-manual/...

    These should be converted to relative paths without the system path prefix.
    """
    pattern = r'\]\(/home/mvillmow/ml-odyssey-manual/([^)]+)\)'
    replacement = r'](\1)'

    new_content, count = re.subn(pattern, replacement, content)
    return new_content, count

def fix_absolute_path_links(content: str, file_path: Path) -> Tuple[str, int]:
    """
    Fix absolute paths like /agents/... to relative paths.

    Calculate the correct relative path based on the file's location.
    """
    # Count slashes in file path to determine directory depth
    # e.g., notes/issues/863/README.md -> depth 3, need ../../../
    depth = len(file_path.parent.parts)
    prefix = '../' * depth if depth > 0 else ''

    # Fix links starting with / (but not //)
    pattern = r'\]\(/(?!/)'
    replacement = f']({prefix}'

    new_content, count = re.subn(pattern, replacement, content)
    return new_content, count

def process_file(file_path: Path, dry_run: bool = False) -> Tuple[int, int]:
    """
    Process a single markdown file to fix invalid links.

    Returns:
        Tuple of (system_path_fixes, absolute_path_fixes)
    """
    try:
        content = file_path.read_text(encoding='utf-8')

        # Fix system path links
        content, system_fixes = fix_system_path_links(content)

        # Fix absolute path links
        content, absolute_fixes = fix_absolute_path_links(content, file_path)

        total_fixes = system_fixes + absolute_fixes

        if total_fixes > 0:
            if dry_run:
                print(f"Would fix {file_path}: {system_fixes} system paths, {absolute_fixes} absolute paths")
            else:
                file_path.write_text(content, encoding='utf-8')
                print(f"Fixed {file_path}: {system_fixes} system paths, {absolute_fixes} absolute paths")

        return system_fixes, absolute_fixes

    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return 0, 0

def main():
    """Main entry point."""
    dry_run = '--dry-run' in sys.argv

    if dry_run:
        print("DRY RUN MODE - No files will be modified\n")

    repo_root = Path(__file__).parent.parent

    # Find all markdown files
    md_files = list(repo_root.glob('**/*.md'))

    # Filter out excluded directories
    excluded_dirs = {'.git', 'node_modules', '.pixi', 'venv', '__pycache__'}
    md_files = [
        f for f in md_files
        if not any(excluded in f.parts for excluded in excluded_dirs)
    ]

    print(f"Found {len(md_files)} markdown files\n")

    total_system_fixes = 0
    total_absolute_fixes = 0
    files_modified = 0

    for md_file in sorted(md_files):
        system_fixes, absolute_fixes = process_file(md_file, dry_run)
        if system_fixes + absolute_fixes > 0:
            files_modified += 1
            total_system_fixes += system_fixes
            total_absolute_fixes += absolute_fixes

    print(f"\n{'Would fix' if dry_run else 'Fixed'} {files_modified} files:")
    print(f"  - System path links: {total_system_fixes}")
    print(f"  - Absolute path links: {total_absolute_fixes}")
    print(f"  - Total fixes: {total_system_fixes + total_absolute_fixes}")

if __name__ == '__main__':
    main()
