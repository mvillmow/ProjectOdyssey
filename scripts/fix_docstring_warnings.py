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
    """DISABLED - Indentation fixes caused issues. Keeping for future reference."""
    # Removing indentation from Args: tags introduced new "unknown argument" warnings
    # This needs a more sophisticated approach that understands docstring context
    return content, 0


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
    # 1. Have 8+ spaces of indentation (inside a docstring section)
    # 2. Don't end with . or ` or :
    # 3. Are followed by either:
    #    - A line with less indentation (next section or closing """)
    #    - The closing """
    # This fixes warnings like "section body should end with a period"

    # Pattern: 8+ spaces + content not ending with . or ` or :, followed by newline and either:
    # - Less than 8 spaces (new section or closing)
    # - End of string
    # - Closing """
    pattern = r"^(        [^:\n]+[^.`:\n])(\n)(?=(?:    \"\"\"|    [A-Z]|\"\"\"|$))"

    matches = len(re.findall(pattern, content, flags=re.MULTILINE))
    fixed_content = re.sub(pattern, r"\1.\2", content, flags=re.MULTILINE)

    return fixed_content, matches


def fix_summary_periods(content: str) -> Tuple[str, int]:
    """Add periods to docstring summary lines (first line of docstring)."""
    # Match docstring opening with summary not ending in period
    # Pattern: """ + text not ending with . + newline (start of body or closing)
    pattern = r'^(    """[^"\n]+[^.\n])(\n)'

    matches = len(re.findall(pattern, content, flags=re.MULTILINE))
    fixed_content = re.sub(pattern, r"\1.\2", content, flags=re.MULTILINE)

    return fixed_content, matches


def fix_multiline_parameter_descriptions(content: str) -> Tuple[str, int]:
    """DISABLED - Multiline fixes need more context awareness."""
    # These warnings need manual review as they require understanding the
    # documentation context to fix properly
    return content, 0


def fix_deprecated_owned(content: str) -> Tuple[str, int]:
    """Replace deprecated 'owned' keyword with 'deinit' or 'var'."""
    fixes = 0

    # Pattern 1: __moveinit__ and __copyinit__ use 'deinit'
    # fn __moveinit__(out self, owned existing: Self) -> fn __moveinit__(out self, deinit existing: Self)
    pattern1 = r"\b(fn __(?:move|copy)init__\([^)]*)(owned)(\s+\w+:\s*Self\))"
    matches1 = len(re.findall(pattern1, content))
    fixed_content = re.sub(pattern1, r"\1deinit\3", content)
    fixes += matches1

    # Pattern 2: Other contexts use 'var'
    # owned param: Type -> var param: Type
    pattern2 = r"\bowned\s+(\w+:)"
    matches2 = len(re.findall(pattern2, fixed_content))
    fixed_content = re.sub(pattern2, r"var \1", fixed_content)
    fixes += matches2

    return fixed_content, fixes


def fix_file(file_path: Path, dry_run: bool = False) -> Tuple[int, int, int, int, int, int]:
    """Fix warnings in a single file.

    Returns (indentation, periods, section_body, summary, multiline_params, deprecated).
    """
    content = file_path.read_text()
    original = content

    # Apply fixes in order
    content, indent_fixes = fix_section_indentation(content)
    content, period_fixes = fix_missing_periods(content)
    content, section_body_fixes = fix_section_body_endings(content)
    content, summary_fixes = fix_summary_periods(content)
    content, multiline_fixes = fix_multiline_parameter_descriptions(content)
    content, deprecated_fixes = fix_deprecated_owned(content)

    if content != original:
        if not dry_run:
            file_path.write_text(content)
        return (
            indent_fixes,
            period_fixes,
            section_body_fixes,
            summary_fixes,
            multiline_fixes,
            deprecated_fixes,
        )

    return 0, 0, 0, 0, 0, 0


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
    total_summary = 0
    total_multiline = 0
    total_deprecated = 0

    for file_path in files:
        if not file_path.exists():
            print(f"⚠️  Skipping {file_path} (not found)")
            continue

        indent, period, section_body, summary, multiline, deprecated = fix_file(file_path, dry_run=args.dry_run)

        total = indent + period + section_body + summary + multiline + deprecated

        if total > 0:
            action = "Would fix" if args.dry_run else "Fixed"
            print(f"{action} {total} warnings in {file_path.relative_to(base_path)}")
            if indent:
                print(f"  - {indent} indentation fixes")
            if period:
                print(f"  - {period} single-line period additions")
            if section_body:
                print(f"  - {section_body} section body endings")
            if summary:
                print(f"  - {summary} summary periods")
            if multiline:
                print(f"  - {multiline} multiline parameter periods")
            if deprecated:
                print(f"  - {deprecated} deprecated 'owned' replacements")

            total_indent += indent
            total_period += period
            total_section_body += section_body
            total_summary += summary
            total_multiline += multiline
            total_deprecated += deprecated

    grand_total = total_indent + total_period + total_section_body + total_summary + total_multiline + total_deprecated

    print(f"\n{'Preview' if args.dry_run else 'Total'}: {grand_total} fixes")
    if total_indent:
        print(f"  - {total_indent} indentation")
    if total_period:
        print(f"  - {total_period} single-line periods")
    if total_section_body:
        print(f"  - {total_section_body} section body endings")
    if total_summary:
        print(f"  - {total_summary} summary periods")
    if total_multiline:
        print(f"  - {total_multiline} multiline parameter periods")
    if total_deprecated:
        print(f"  - {total_deprecated} deprecated 'owned' replacements")

    return 0


if __name__ == "__main__":
    exit(main())
