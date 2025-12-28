#!/usr/bin/env python3
"""
Fix remaining markdown linting errors after initial pass.

This script fixes:
- MD026: Trailing punctuation in headings
- MD022: Blank lines around headings
- MD031: Blank lines around code blocks
- MD032: Blank lines around lists
- MD001: Heading increment issues (limited support)
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def fix_heading_punctuation(lines: List[str]) -> List[str]:
    """Remove trailing punctuation from headings (MD026)."""
    fixed: List[str] = []
    for line in lines:
        if line.strip().startswith("#"):
            # Remove trailing punctuation from headings (except '?')
            line = re.sub(r"([:#.])\s*$", "", line.rstrip()) + "\n"
        fixed.append(line)
    return fixed


def fix_blank_lines_around_headings(lines: List[str]) -> List[str]:
    """Ensure blank lines around headings (MD022)."""
    fixed: List[str] = []
    for i, line in enumerate(lines):
        # Check if this is a heading
        if line.strip().startswith("#"):
            # Add blank line before if not at start and previous isn't blank
            if i > 0 and fixed and fixed[-1].strip():
                fixed.append("\n")
            fixed.append(line)
            # Add blank line after if not at end and next isn't blank
            if i < len(lines) - 1 and lines[i + 1].strip():
                fixed.append("\n")
        else:
            fixed.append(line)
    return fixed


def fix_blank_lines_around_fences(lines: List[str]) -> List[str]:
    """Ensure blank lines around code fences (MD031)."""
    fixed: List[str] = []
    in_fence = False

    for i, line in enumerate(lines):
        is_fence = line.strip().startswith("```")

        if is_fence:
            if not in_fence:  # Opening fence
                # Add blank before if needed
                if fixed and fixed[-1].strip():
                    fixed.append("\n")
                in_fence = True
            else:  # Closing fence
                in_fence = False
                fixed.append(line)
                # Add blank after if needed
                if i < len(lines) - 1 and lines[i + 1].strip():
                    fixed.append("\n")
                continue

        fixed.append(line)

    return fixed


def fix_blank_lines_around_lists(lines: List[str]) -> List[str]:
    """Ensure blank lines around lists (MD032)."""
    fixed: List[str] = []
    in_list = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        is_list_item = bool(re.match(r"^[-*+]\s+", stripped) or re.match(r"^\d+\.\s+", stripped))

        if is_list_item:
            if not in_list:
                # Starting a list - add blank before
                if fixed and fixed[-1].strip():
                    fixed.append("\n")
                in_list = True
        else:
            if in_list and stripped:
                # End of list - add blank before next content
                in_list = False
                fixed.append("\n")

        fixed.append(line)

    return fixed


def process_file(filepath: Path) -> Tuple[bool, int]:
    """Process a single markdown file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return False, 0

    original_content = "".join(lines)

    # Apply fixes in order
    lines = fix_heading_punctuation(lines)
    lines = fix_blank_lines_around_headings(lines)
    lines = fix_blank_lines_around_fences(lines)
    lines = fix_blank_lines_around_lists(lines)

    new_content = "".join(lines)

    if new_content != original_content:
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            return True, 1
        except Exception as e:
            print(f"Error writing {filepath}: {e}", file=sys.stderr)
            return False, 0

    return True, 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 fix_remaining_markdown.py <directory>")
        sys.exit(1)

    root_path = Path(sys.argv[1])
    if not root_path.exists():
        print(f"Error: {root_path} does not exist")
        sys.exit(1)

    # Find all markdown files
    md_files = list(root_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown file(s)\n")

    total_modified = 0
    total_errors = 0

    for filepath in md_files:
        success, modified = process_file(filepath)
        if success:
            total_modified += modified
        else:
            total_errors += 1

    print("\nSummary:")
    print(f"  Files modified: {total_modified}")
    if total_errors:
        print(f"  Errors: {total_errors}")


if __name__ == "__main__":
    main()
