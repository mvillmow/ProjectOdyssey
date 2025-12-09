#!/usr/bin/env python3
"""
Fix Mojo docstring warnings by ensuring all docstring summaries
and parameter/return descriptions end with periods or backticks.
"""

import re
import sys
from pathlib import Path


def fix_docstring_summary(content: str) -> str:
    """Fix docstring summaries to end with periods."""
    # Pattern: """SomeSummary\n or """SomeSummary
    # Should end with period if it doesn't end with period, backtick, or colon

    # Match triple-quote docstrings (single line summaries)
    pattern = r'(""")([^"\n]+?)(?<![.`:\)])(\n)'

    def repl(match):
        quote = match.group(1)
        summary = match.group(2).strip()
        newline = match.group(3)

        # Don't add period if it already ends with punctuation
        if summary and not summary[-1] in '.`!?:)':
            return f'{quote}{summary}.{newline}'
        return match.group(0)

    content = re.sub(pattern, repl, content)

    return content


def fix_param_descriptions(content: str) -> str:
    """Fix parameter and return descriptions to end with periods."""
    # Pattern for parameter descriptions like:
    #     tensor: Input tensor
    # Should be:
    #     tensor: Input tensor.

    lines = content.split('\n')
    result = []
    in_docstring = False

    for line in lines:
        # Track if we're in a docstring
        if '"""' in line:
            in_docstring = not in_docstring

        # Fix parameter/return descriptions in docstrings
        if in_docstring and ': ' in line:
            # Match lines like "    param: Description"
            match = re.match(r'(\s+)(\w+):\s+(.+?)(?<![.`!?)])(\s*)$', line)
            if match:
                indent = match.group(1)
                param = match.group(2)
                desc = match.group(3)
                trailing = match.group(4)

                # Add period if missing
                if desc and not desc[-1] in '.`!?:)':
                    line = f'{indent}{param}: {desc}.{trailing}'

        result.append(line)

    return '\n'.join(result)


def fix_section_bodies(content: str) -> str:
    """Fix section bodies (Returns:, Raises:, etc.) to end with periods."""
    lines = content.split('\n')
    result = []
    in_docstring = False
    in_section = False

    for i, line in enumerate(lines):
        # Track if we're in a docstring
        if '"""' in line:
            in_docstring = not in_docstring
            in_section = False

        # Check if we're starting a section
        if in_docstring and re.match(r'\s+(Returns?|Raises?|Args?|Notes?):\s*$', line):
            in_section = True

        # Fix section body descriptions
        if in_section and line.strip() and not line.strip().endswith(':'):
            # Check if it's a description line (indented more than section header)
            if re.match(r'\s{8,}(.+?)(?<![.`!?)])(\s*)$', line):
                match = re.match(r'(\s+)(.+?)(?<![.`!?)])(\s*)$', line)
                if match:
                    indent = match.group(1)
                    desc = match.group(2)
                    trailing = match.group(3)

                    # Add period if missing and it's not a parameter description
                    if desc and not ':' in desc and not desc[-1] in '.`!?:)':
                        line = f'{indent}{desc}.{trailing}'

        result.append(line)

    return '\n'.join(result)


def fix_file(filepath: Path) -> bool:
    """Fix docstrings in a single file. Returns True if changes were made."""
    try:
        content = filepath.read_text(encoding='utf-8')
        original = content

        # Apply fixes
        content = fix_docstring_summary(content)
        content = fix_param_descriptions(content)
        content = fix_section_bodies(content)

        if content != original:
            filepath.write_text(content, encoding='utf-8')
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return False


def main():
    """Fix docstrings in all Mojo files."""
    shared_dir = Path("shared")
    if not shared_dir.exists():
        print("Error: shared/ directory not found", file=sys.stderr)
        return 1

    mojo_files = list(shared_dir.rglob("*.mojo"))
    print(f"Found {len(mojo_files)} .mojo files")

    fixed_count = 0
    for filepath in mojo_files:
        if fix_file(filepath):
            print(f"Fixed: {filepath}")
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
