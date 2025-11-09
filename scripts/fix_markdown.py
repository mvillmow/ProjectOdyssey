#!/usr/bin/env python3
"""
Fix markdown linting errors systematically.

This script fixes common markdown linting issues:
- MD022: Headings should be surrounded by blank lines
- MD031: Code blocks should be surrounded by blank lines
- MD032: Lists should be surrounded by blank lines
- MD040: Code blocks should have language specified
- MD012: Multiple consecutive blank lines
"""

import re
import sys
from pathlib import Path
from typing import List


def fix_markdown_file(file_path: Path) -> bool:
    """
    Fix markdown linting errors in a file.

    Args:
        file_path: Path to the markdown file

    Returns:
        True if file was modified, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return False

    original_content = content
    lines = content.split('\n')
    new_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Get previous and next lines (if they exist)
        prev_line = lines[i-1] if i > 0 else ''
        next_line = lines[i+1] if i < len(lines) - 1 else ''

        # MD012: Remove multiple consecutive blank lines
        if line == '' and prev_line == '' and new_lines and new_lines[-1] == '':
            i += 1
            continue

        # MD022: Headings should be surrounded by blank lines
        if line.startswith('#'):
            # Add blank line before heading if needed
            if new_lines and new_lines[-1] != '':
                new_lines.append('')
            new_lines.append(line)
            # Add blank line after heading if needed
            if next_line and next_line != '' and not next_line.startswith('#'):
                new_lines.append('')
            i += 1
            continue

        # MD031/MD040: Code fences should be surrounded by blank lines and have language
        if line.startswith('```'):
            # Add blank line before fence if needed
            if new_lines and new_lines[-1] != '':
                new_lines.append('')

            # MD040: Add language if missing (use 'text' as default)
            if line.strip() == '```':
                # Try to infer language from context or use 'text'
                new_lines.append('```text')
            else:
                new_lines.append(line)

            # Copy everything until closing fence
            i += 1
            while i < len(lines):
                fence_line = lines[i]
                new_lines.append(fence_line)
                if fence_line.startswith('```'):
                    # Add blank line after closing fence if needed
                    next_after_fence = lines[i+1] if i+1 < len(lines) else ''
                    if next_after_fence and next_after_fence != '':
                        new_lines.append('')
                    i += 1
                    break
                i += 1
            continue

        # MD032: Lists should be surrounded by blank lines
        list_pattern = r'^(\s*)[-*+]|\d+\.'
        if re.match(list_pattern, line):
            # Add blank line before list if needed
            if new_lines and new_lines[-1] != '' and not re.match(list_pattern, prev_line):
                new_lines.append('')

            # Copy all list items
            while i < len(lines):
                curr_line = lines[i]
                # Check if still in list (list item or continuation)
                if re.match(list_pattern, curr_line) or (curr_line.startswith('  ') and prev_line):
                    new_lines.append(curr_line)
                    prev_line = curr_line
                    i += 1
                else:
                    break

            # Add blank line after list if needed
            if i < len(lines) and lines[i] != '':
                new_lines.append('')
            continue

        # Default: just copy the line
        new_lines.append(line)
        i += 1

    # Join lines back
    new_content = '\n'.join(new_lines)

    # Final cleanup: ensure file ends with single newline
    if new_content and not new_content.endswith('\n'):
        new_content += '\n'

    # Write back if changed
    if new_content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Fixed: {file_path}")
            return True
        except Exception as e:
            print(f"Error writing {file_path}: {e}", file=sys.stderr)
            return False

    return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python fix_markdown.py <file_or_directory>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if not path.exists():
        print(f"Error: {path} does not exist", file=sys.stderr)
        sys.exit(1)

    files_to_fix = []
    if path.is_file():
        if path.suffix == '.md':
            files_to_fix.append(path)
    else:
        files_to_fix = list(path.rglob('*.md'))

    fixed_count = 0
    for file_path in files_to_fix:
        if fix_markdown_file(file_path):
            fixed_count += 1

    print(f"\nFixed {fixed_count} of {len(files_to_fix)} files")


if __name__ == '__main__':
    main()
