#!/usr/bin/env python3
"""
Automatically fix common markdown linting errors.

Fixes:
- MD040: Add language tags to code blocks
- MD022: Add blank lines around headings
- MD031/MD032: Add blank lines around code blocks and lists
- MD034: Convert bare URLs to markdown links
"""

import re
import sys
from pathlib import Path


def fix_code_block_languages(content: str) -> str:
    """Add 'text' language to unmarked code blocks."""
    # Match code blocks without language: ```\n
    pattern = r"```\n"
    replacement = "```text\n"
    return content.replace(pattern, replacement)


def fix_blank_lines_around_headings(content: str) -> str:
    """Add blank lines before and after headings."""
    lines = content.split("\n")
    result = []

    for i, line in enumerate(lines):
        # Check if current line is a heading
        if line.startswith("#") and not line.startswith("#!"):
            # Add blank line before heading if previous line is not blank
            if i > 0 and result and result[-1].strip():
                result.append("")

            result.append(line)

            # Add blank line after heading if next line is not blank
            if i < len(lines) - 1 and lines[i + 1].strip():
                result.append("")
        else:
            result.append(line)

    return "\n".join(result)


def fix_blank_lines_around_code_blocks(content: str) -> str:
    """Add blank lines before and after code blocks."""
    lines = content.split("\n")
    result = []
    in_code_block = False

    for i, line in enumerate(lines):
        if line.startswith("```"):
            if not in_code_block:
                # Starting code block - add blank line before if needed
                if result and result[-1].strip():
                    result.append("")
                in_code_block = True
            else:
                # Ending code block
                in_code_block = False

            result.append(line)

            # Add blank line after closing ``` if next line is not blank
            if not in_code_block and i < len(lines) - 1 and lines[i + 1].strip():
                result.append("")
        else:
            result.append(line)

    return "\n".join(result)


def fix_blank_lines_around_lists(content: str) -> str:
    """Add blank lines before and after lists."""
    lines = content.split("\n")
    result = []
    in_list = False

    for i, line in enumerate(lines):
        is_list_item = bool(re.match(r"^\s*[-*+]\s+", line) or re.match(r"^\s*\d+\.\s+", line))

        if is_list_item:
            if not in_list:
                # Starting list - add blank line before if needed
                if result and result[-1].strip():
                    result.append("")
                in_list = True
            result.append(line)
        else:
            if in_list and line.strip():
                # Ending list - add blank line after if next line has content
                result.append("")
                in_list = False
            result.append(line)

    return "\n".join(result)


def fix_bare_urls(content: str) -> str:
    """Convert bare URLs to markdown links."""

    # Only fix URLs that are not already in markdown link syntax
    # Pattern: URL not preceded by ]( and not followed by )
    def replace_url(match):
        url = match.group(0)
        # Check if URL is already in a markdown link
        start = match.start()
        if start >= 2 and content[start - 2 : start] == "](":
            return url
        return f"<{url}>"

    # Match http/https URLs
    pattern = r"https?://[a-zA-Z0-9][-a-zA-Z0-9@:%._\+~#=]{0,256}\.[a-zA-Z0-9]{1,6}\b[-a-zA-Z0-9@:%_\+.~#?&/=]*"

    return re.sub(pattern, replace_url, content)


def fix_file(file_path: Path) -> bool:
    """Fix markdown linting issues in a file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        original = content

        # Apply fixes in order
        content = fix_code_block_languages(content)
        content = fix_blank_lines_around_headings(content)
        content = fix_blank_lines_around_code_blocks(content)
        content = fix_blank_lines_around_lists(content)
        content = fix_bare_urls(content)

        # Write back only if changed
        if content != original:
            file_path.write_text(content, encoding="utf-8")
            print(f"Fixed: {file_path}")
            return True

        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}", file=sys.stderr)
        return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: fix_markdown_lint.py <file1.md> [file2.md ...]")
        return 1

    fixed_count = 0
    for file_path_str in sys.argv[1:]:
        file_path = Path(file_path_str)
        if file_path.suffix == ".md":
            if fix_file(file_path):
                fixed_count += 1

    print(f"\nFixed {fixed_count} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
