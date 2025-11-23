#!/usr/bin/env python3
"""
Fix List initialization pattern in test files.

Replaces:
    var shape = List[Int]()
    shape[0] = value

With:
    var shape = List[Int]()
    shape.append(value)
"""

import re
import sys
from pathlib import Path


def fix_list_initialization(file_path: Path) -> tuple[int, bool]:
    """
    Fix List initialization pattern in a single file.

    Returns:
        (num_fixes, was_modified)
    """
    content = file_path.read_text()
    original_content = content

    # Pattern: var shape = List[Int]() followed by shape[N] = value
    # We need to find consecutive lines where shape[N] = value

    # Split into lines for processing
    lines = content.split('\n')
    modified_lines = []
    fixes = 0

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this line creates a List
        list_match = re.match(r'(\s+)var (\w+) = List\[Int\]\(\)\s*$', line)

        if list_match:
            indent = list_match.group(1)
            var_name = list_match.group(2)
            modified_lines.append(line)
            i += 1

            # Look ahead for pattern: var_name[N] = value
            while i < len(lines):
                next_line = lines[i]
                index_match = re.match(rf'{indent}{var_name}\[(\d+)\] = (.+)$', next_line)

                if index_match:
                    index = index_match.group(1)
                    value = index_match.group(2)
                    # Replace with append
                    modified_lines.append(f'{indent}{var_name}.append({value})')
                    fixes += 1
                    i += 1
                else:
                    # No more index assignments
                    break
        else:
            modified_lines.append(line)
            i += 1

    new_content = '\n'.join(modified_lines)
    was_modified = new_content != original_content

    if was_modified:
        file_path.write_text(new_content)

    return fixes, was_modified


def main():
    """Fix all test files."""
    test_dir = Path('/home/mvillmow/ml-odyssey/tests')

    # Find all .mojo files
    mojo_files = list(test_dir.rglob('*.mojo'))

    total_fixes = 0
    modified_files = []

    for file_path in sorted(mojo_files):
        fixes, was_modified = fix_list_initialization(file_path)

        if was_modified:
            print(f"✅ {file_path.relative_to(test_dir.parent)}: {fixes} fixes")
            modified_files.append(file_path)
            total_fixes += fixes

    print(f"\n📊 Summary:")
    print(f"   Total files modified: {len(modified_files)}")
    print(f"   Total fixes: {total_fixes}")

    if modified_files:
        print(f"\n📝 Modified files:")
        for file_path in modified_files:
            print(f"   - {file_path.relative_to(test_dir.parent)}")


if __name__ == '__main__':
    main()
