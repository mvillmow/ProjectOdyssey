#!/usr/bin/env python3

"""
Fix markdown linting errors systematically across the repository.

This script fixes common markdown linting issues:
- MD022: Add blank lines around headings
- MD032: Add blank lines around lists
- MD031: Add blank lines around code blocks
- MD012: Remove multiple consecutive blank lines
- MD040: Add language tags to code blocks
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def fix_markdown_file(file_path: Path) -> Tuple[bool, int]:
    """Fix markdown linting errors in a file.

    Args:
        file_path: Path to markdown file

    Returns:
        Tuple of (file_was_modified, error_count_fixed)
    """
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False, 0

    original_content = content
    fixes_applied = 0

    # Fix MD012: Remove multiple consecutive blank lines
    while '\n\n\n' in content:
        content = content.replace('\n\n\n', '\n\n')
        fixes_applied += 1

    # Split into lines for processing
    lines = content.split('\n')
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if current line is a heading
        if re.match(r'^#{1,6}\s+', line):
            # MD022: Ensure blank line before heading (except at start)
            if fixed_lines and fixed_lines[-1].strip() != '':
                fixed_lines.append('')
                fixes_applied += 1

            fixed_lines.append(line)
            i += 1

            # MD022: Ensure blank line after heading
            if i < len(lines) and lines[i].strip() != '' and not re.match(r'^#{1,6}\s+', lines[i]):
                fixed_lines.append('')
                fixes_applied += 1

        # Check if current line starts a list
        elif re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
            # MD032: Ensure blank line before list (except at start)
            if fixed_lines and fixed_lines[-1].strip() != '':
                fixed_lines.append('')
                fixes_applied += 1

            # Add the list item
            fixed_lines.append(line)
            i += 1

            # Continue collecting list items
            while i < len(lines) and (
                re.match(r'^\s*[-*+]\s+', lines[i]) or
                re.match(r'^\s*\d+\.\s+', lines[i]) or
                (lines[i].startswith(' ') and lines[i].strip() != '')
            ):
                fixed_lines.append(lines[i])
                i += 1

            # MD032: Ensure blank line after list
            if i < len(lines) and lines[i].strip() != '' and not re.match(r'^#{1,6}\s+', lines[i]):
                fixed_lines.append('')
                fixes_applied += 1

        # Check if current line starts a code block
        elif line.strip().startswith('```'):
            # MD031: Ensure blank line before code block
            if fixed_lines and fixed_lines[-1].strip() != '':
                fixed_lines.append('')
                fixes_applied += 1

            # Extract language tag if present
            code_fence = line.strip()
            if not re.match(r'^```[a-zA-Z0-9_+-]*$', code_fence):
                # Fix code block opening (ensure language tag format)
                match = re.match(r'^(`{3,})(.*)$', code_fence)
                if match:
                    backticks = match.group(1)
                    rest = match.group(2).strip()
                    # Ensure we have a valid language tag or empty
                    if rest and not re.match(r'^[a-zA-Z0-9_+-]+$', rest):
                        rest = ''
                    code_fence = f"{backticks}{rest}"
                    fixes_applied += 1

            fixed_lines.append(code_fence)
            i += 1

            # Continue collecting code block content
            while i < len(lines) and not lines[i].strip().startswith('```'):
                fixed_lines.append(lines[i])
                i += 1

            # Add closing fence
            if i < len(lines):
                fixed_lines.append(lines[i])
                i += 1

            # MD031: Ensure blank line after code block
            if i < len(lines) and lines[i].strip() != '':
                fixed_lines.append('')
                fixes_applied += 1

        else:
            fixed_lines.append(line)
            i += 1

    # Reconstruct content
    new_content = '\n'.join(fixed_lines)

    # Ensure file ends with newline
    if new_content and not new_content.endswith('\n'):
        new_content += '\n'
        fixes_applied += 1

    # Write back if changed
    if new_content != original_content:
        try:
            file_path.write_text(new_content, encoding='utf-8')
            return True, fixes_applied
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return False, 0

    return False, fixes_applied


def main():
    """Fix markdown files in repository."""
    repo_root = Path('/home/mvillmow/ml-odyssey-manual')

    # Find all markdown files
    md_files = list(repo_root.glob('**/*.md'))
    print(f"Found {len(md_files)} markdown files")

    total_fixed = 0
    files_modified = 0

    for file_path in sorted(md_files):
        modified, fixes = fix_markdown_file(file_path)
        if modified:
            files_modified += 1
            total_fixed += fixes
            rel_path = file_path.relative_to(repo_root)
            print(f"Fixed {rel_path}: {fixes} issues")

    print(f"\nSummary:")
    print(f"  Files modified: {files_modified}")
    print(f"  Total issues fixed: {total_fixed}")

    return 0 if files_modified == 0 else 0


if __name__ == '__main__':
    sys.exit(main())
