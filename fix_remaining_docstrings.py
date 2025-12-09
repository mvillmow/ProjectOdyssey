#!/usr/bin/env python3
"""
Fix remaining Mojo docstring warnings by adding periods to lines ending with ).
"""

import re
import sys
from pathlib import Path


def fix_paren_endings(filepath: Path) -> bool:
    """Fix descriptions that end with ) to have a period after the )."""
    try:
        content = filepath.read_text(encoding='utf-8')
        lines = content.split('\n')
        result = []
        modified = False
        in_docstring = False

        for i, line in enumerate(lines):
            original_line = line

            # Track docstring state
            if '"""' in line:
                in_docstring = not in_docstring

            # Fix parameter descriptions that end with )
            if in_docstring and ': ' in line and line.rstrip().endswith(')'):
                # Match parameter descriptions like "param: Description (context)"
                match = re.match(r'(\s+\w+:\s+.+\))(\s*)$', line)
                if match and not line.rstrip().endswith(').'):
                    line = match.group(1) + '.' + match.group(2)
                    modified = True

            # Fix example lines that end with )
            if in_docstring and line.strip() and not ':' in line.split()[-1] if line.split() else False:
                if line.rstrip().endswith(')') and not line.rstrip().endswith(').'):
                    # Check if it's not already followed by a blank line or """
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line and not next_line.startswith('"""'):
                            # Add period
                            line = line.rstrip() + '.'
                            modified = True

            result.append(line)

        if modified:
            filepath.write_text('\n'.join(result), encoding='utf-8')
            return True
        return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return False


def main():
    """Fix remaining docstring warnings."""
    # Target the specific files with warnings
    files_to_fix = [
        "shared/autograd/functional.mojo",
        "shared/autograd/grad_utils.mojo",
    ]

    fixed_count = 0
    for filepath_str in files_to_fix:
        filepath = Path(filepath_str)
        if not filepath.exists():
            print(f"Warning: {filepath} not found", file=sys.stderr)
            continue

        if fix_paren_endings(filepath):
            print(f"Fixed: {filepath}")
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
