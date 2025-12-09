#!/usr/bin/env python3
"""
Comprehensively fix ALL Mojo docstring warnings.
"""

import re
import sys
from pathlib import Path


def fix_file_docstrings(filepath: Path) -> bool:
    """Fix all docstring issues in a file."""
    try:
        content = filepath.read_text(encoding='utf-8')
        original = content
        lines = content.split('\n')
        result = []
        in_docstring = False
        docstring_start_line = -1

        for i, line in enumerate(lines):
            # Track when we enter/exit docstrings
            triple_quote_count = line.count('"""')
            if triple_quote_count > 0:
                if triple_quote_count == 2 and line.strip().startswith('"""') and line.strip().endswith('"""'):
                    # Single-line docstring
                    in_docstring = False
                elif not in_docstring:
                    in_docstring = True
                    docstring_start_line = i
                else:
                    in_docstring = False

            if in_docstring or '"""' in line:
                # Fix various patterns

                # Pattern 1: First line of docstring (summary) must end with period
                if line.strip().startswith('"""') and not line.strip().endswith('"""'):
                    # Multi-line docstring starting
                    next_line_idx = i + 1
                    if next_line_idx < len(lines):
                        summary_line = lines[next_line_idx].strip()
                        if summary_line and not summary_line.endswith(('.', '`', ':', '!')):
                            # Summary needs a period
                            indent = len(lines[next_line_idx]) - len(lines[next_line_idx].lstrip())
                            lines[next_line_idx] = ' ' * indent + summary_line + '.'

                # Pattern 2: Parameter descriptions (lines with ": ")
                if ': ' in line and not line.strip().startswith('#'):
                    stripped = line.rstrip()
                    if stripped and not stripped[-1] in ('.', '`'):
                        line = stripped + '.'

                # Pattern 3: Section bodies that don't end with period or backtick
                # (But not section headers like "Returns:" or lines with ":")
                if line.strip() and not line.strip().endswith((':',)) and ': ' not in line[-20:] if len(line) >= 20 else True:
                    stripped = line.rstrip()
                    # If it's indented content (not a header) and doesn't end properly
                    if len(line) - len(line.lstrip()) >= 8 and stripped:
                        if not stripped[-1] in ('.', '`', ':', ')'):
                            line = stripped + '.'

            result.append(line)

        # Second pass: Fix any remaining issues
        content = '\n'.join(result)
        lines = content.split('\n')
        result = []
        in_docstring = False

        for i, line in enumerate(lines):
            if '"""' in line:
                in_docstring = not in_docstring

            # Additional fix: Lines in docstrings ending with specific problematic patterns
            if in_docstring and line.strip():
                # Fix lines ending with ) but no period
                if line.rstrip().endswith(')') and ': ' in line:
                    if not line.rstrip().endswith(').'):
                        line = line.rstrip() + '.'

                # Fix lines ending with ] but no period (for array notations)
                if line.rstrip().endswith(']') and not line.rstrip().endswith('].'):
                    # Only if it's a description line, not code
                    if ': ' in line or (i > 0 and any(keyword in lines[i-1] for keyword in ['Returns:', 'Raises:', 'Examples:'])):
                        line = line.rstrip() + '.'

                # Fix lines ending with numbers or letters (unlikely to be intentional)
                if line.rstrip() and line.rstrip()[-1].isalnum():
                    # Check if it's a parameter description or return description
                    if ': ' in line or (i > 0 and 'Returns:' in lines[i-1]):
                        if not line.rstrip().endswith('.'):
                            line = line.rstrip() + '.'

            result.append(line)

        new_content = '\n'.join(result)

        if new_content != original:
            filepath.write_text(new_content, encoding='utf-8')
            return True
        return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    """Fix all docstring warnings in all Mojo files."""
    shared_dir = Path("shared")
    if not shared_dir.exists():
        print("Error: shared/ directory not found", file=sys.stderr)
        return 1

    mojo_files = list(shared_dir.rglob("*.mojo"))
    print(f"Found {len(mojo_files)} .mojo files")

    fixed_count = 0
    for filepath in mojo_files:
        if fix_file_docstrings(filepath):
            print(f"Fixed: {filepath}")
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
