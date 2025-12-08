#!/usr/bin/env python3
"""Fix Mojo docstring formatting warnings.

Usage:
    python3 scripts/fix_docstring_warnings.py --file <path>  # Fix one file
    python3 scripts/fix_docstring_warnings.py --all          # Fix all files
    python3 scripts/fix_docstring_warnings.py --dry-run      # Preview changes
    python3 scripts/fix_docstring_warnings.py --top-10       # Fix top 10 files
"""

import argparse
import re
from pathlib import Path
from typing import Tuple


def fix_section_indentation(content: str) -> Tuple[str, int]:
    """Remove 4-space indentation from Args/Returns/Raises tags."""
    patterns = [
        (r"^(    )(Args:)$", r"\2"),
        (r"^(    )(Returns:)$", r"\2"),
        (r"^(    )(Raises:)$", r"\2"),
        (r"^(    )(Examples:)$", r"\2"),
        (r"^(    )(Note:)$", r"\2"),
    ]

    fixed_content = content
    fixes = 0

    for pattern, replacement in patterns:
        new_content = re.sub(pattern, replacement, fixed_content, flags=re.MULTILINE)
        if new_content != fixed_content:
            fixes += new_content.count(replacement) - fixed_content.count(replacement)
            fixed_content = new_content

    return fixed_content, fixes


def fix_missing_periods(content: str) -> Tuple[str, int]:
    """Add periods to parameter descriptions."""
    # Match parameter descriptions (8 spaces) not ending with . or `
    pattern = r"^(        \w+:.*[^.`\n])(\n)"

    matches = len(re.findall(pattern, content, flags=re.MULTILINE))
    fixed_content = re.sub(pattern, r"\1.\2", content, flags=re.MULTILINE)

    return fixed_content, matches


def fix_section_body_endings(content: str) -> Tuple[str, int]:
    """Add periods to section body endings (Note, Examples, Returns multiline descriptions)."""
    # Match lines that:
    # 1. Have 8 spaces of indentation (inside a docstring section)
    # 2. Don't end with . or ` or :
    # 3. Are followed by either:
    #    - A line with less indentation (next section or closing """)
    #    - The closing """
    # This fixes warnings like "section body should end with a period"

    # Pattern: 8 spaces + content not ending with . or ` or :, followed by newline and either:
    # - Less than 8 spaces (new section or closing)
    # - End of string
    # - Closing """
    pattern = r"^(        [^:\n]+[^.`:\n])(\n)(?=(?:    \"\"\"|    [A-Z]|\"\"\"|$))"

    matches = len(re.findall(pattern, content, flags=re.MULTILINE))
    fixed_content = re.sub(pattern, r"\1.\2", content, flags=re.MULTILINE)

    return fixed_content, matches


def fix_file(file_path: Path, dry_run: bool = False) -> Tuple[int, int, int]:
    """Fix warnings in a single file. Returns (indentation_fixes, period_fixes, section_body_fixes)."""
    content = file_path.read_text()
    original = content

    # Apply fixes
    content, indent_fixes = fix_section_indentation(content)
    content, period_fixes = fix_missing_periods(content)
    content, section_body_fixes = fix_section_body_endings(content)

    if content != original:
        if not dry_run:
            file_path.write_text(content)
        return indent_fixes, period_fixes, section_body_fixes

    return 0, 0, 0


# Top 10 files by warning count (from warnings.log analysis)
TOP_10_FILES = [
    "shared/core/shape.mojo",  # 63 warnings
    "shared/core/elementwise.mojo",  # 62 warnings
    "shared/core/extensor.mojo",  # 58 warnings
    "shared/utils/visualization.mojo",  # 52 warnings
    "shared/core/conv.mojo",  # 52 warnings
    "shared/core/activation.mojo",  # 49 warnings
    "shared/utils/config.mojo",  # 46 warnings
    "shared/core/normalization.mojo",  # 38 warnings
    "shared/testing/models.mojo",  # 36 warnings
    "shared/core/loss.mojo",  # 34 warnings
]


def main():
    parser = argparse.ArgumentParser(description="Fix Mojo docstring warnings")
    parser.add_argument("--file", type=Path, help="Fix specific file")
    parser.add_argument("--all", action="store_true", help="Fix all .mojo files")
    parser.add_argument("--top-10", action="store_true", help="Fix top 10 files")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes")

    args = parser.parse_args()

    base_path = Path("/home/mvillmow/ml-odyssey")

    if args.file:
        files = [args.file]
    elif args.top_10:
        files = [base_path / f for f in TOP_10_FILES]
    elif args.all:
        files = list((base_path / "shared").rglob("*.mojo"))
    else:
        print("Error: Specify --file, --all, or --top-10")
        return 1

    total_indent = 0
    total_period = 0
    total_section_body = 0

    for file_path in files:
        if not file_path.exists():
            print(f"⚠️  Skipping {file_path} (not found)")
            continue

        indent, period, section_body = fix_file(file_path, dry_run=args.dry_run)

        if indent + period + section_body > 0:
            action = "Would fix" if args.dry_run else "Fixed"
            print(f"{action} {indent + period + section_body} warnings in {file_path.relative_to(base_path)}")
            print(f"  - {indent} indentation fixes")
            print(f"  - {period} period additions")
            print(f"  - {section_body} section body endings")
            total_indent += indent
            total_period += period
            total_section_body += section_body

    print(f"\n{'Preview' if args.dry_run else 'Total'}: {total_indent + total_period + total_section_body} fixes")
    print(f"  - {total_indent} indentation")
    print(f"  - {total_period} periods")
    print(f"  - {total_section_body} section body endings")

    return 0


if __name__ == "__main__":
    exit(main())
