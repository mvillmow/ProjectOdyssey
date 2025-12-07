#!/usr/bin/env python3
"""
Unified markdown linting fixer.

This script automatically fixes common markdown linting issues identified by markdownlint-cli2:
- MD012: Remove multiple consecutive blank lines
- MD022: Add blank lines around headings
- MD026: Remove trailing punctuation from headings
- MD029: Fix ordered list numbering (use 1. for all items)
- MD031: Add blank lines around code blocks
- MD032: Add blank lines around lists
- MD036: Convert bold text used as headings to actual headings
- MD040: Add language tags to code blocks

Usage:
    python scripts/fix_markdown.py <file_or_directory> [options]

Examples:
    # Fix a single file
    python scripts/fix_markdown.py README.md

    # Fix all markdown files in a directory
    python scripts/fix_markdown.py notes/

    # Fix all markdown files in repository
    python scripts/fix_markdown.py .

    # Dry run (show what would be fixed without making changes)
    python scripts/fix_markdown.py . --dry-run

    # Verbose output
    python scripts/fix_markdown.py . --verbose
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Tuple


class MarkdownFixer:
    """Fixes common markdown linting issues."""

    def __init__(self, verbose: bool = False, dry_run: bool = False):
        """
        Initialize the markdown fixer.

        Args:
            verbose: Enable verbose output
            dry_run: Show what would be fixed without making changes
        """
        self.verbose = verbose
        self.dry_run = dry_run
        self.fixes_applied = 0

    def fix_file(self, file_path: Path) -> Tuple[bool, int]:
        """
        Fix markdown linting errors in a file.

        Args:
            file_path: Path to markdown file

        Returns:
            Tuple of (file_was_modified, error_count_fixed).
        """
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)
            return False, 0

        original_content = content
        fixes = 0

        # Apply fixes in order
        content, fix_count = self._fix_md012_multiple_blank_lines(content)
        fixes += fix_count

        content, fix_count = self._fix_md040_code_language(content)
        fixes += fix_count

        content, fix_count = self._fix_md026_heading_punctuation(content)
        fixes += fix_count

        content, fix_count = self._fix_structural_issues(content)
        fixes += fix_count

        # Ensure file ends with single newline
        if content and not content.endswith("\n"):
            content += "\n"
            fixes += 1

        # Write back if changed
        if content != original_content:
            if self.dry_run:
                print(f"[DRY RUN] Would fix {file_path}: {fixes} issues")
                return True, fixes

            try:
                file_path.write_text(content, encoding="utf-8")
                if self.verbose:
                    print(f"Fixed {file_path}: {fixes} issues")
                return True, fixes
            except Exception as e:
                print(f"Error writing {file_path}: {e}", file=sys.stderr)
                return False, 0

        if self.verbose:
            print(f"No changes needed for {file_path}")
        return False, 0

    def _fix_md012_multiple_blank_lines(self, content: str) -> Tuple[str, int]:
        """Fix MD012: Remove multiple consecutive blank lines."""
        fixes = 0
        while "\n\n\n" in content:
            content = content.replace("\n\n\n", "\n\n")
            fixes += 1
        return content, fixes

    def _fix_md040_code_language(self, content: str) -> Tuple[str, int]:
        """Fix MD040: Add language tags to code blocks."""
        fixes = 0
        # Find ``` without a language tag
        new_content = re.sub(
            r"^```\s*\n",  # ``` followed by optional whitespace and newline
            "```text\n",  # Add 'text' language tag
            content,
            flags=re.MULTILINE,
        )
        if new_content != content:
            fixes = content.count("```\n") - new_content.count("```\n")
        return new_content, fixes

    def _fix_md026_heading_punctuation(self, content: str) -> Tuple[str, int]:
        """Fix MD026: Remove trailing punctuation from headings."""
        fixes = 0
        lines = content.split("\n")
        fixed_lines = []

        for line in lines:
            # Remove trailing colons, periods, etc. from headings
            if re.match(r"^#{1,6}\s+", line):
                original_line = line
                line = re.sub(r"[:.,;!?]+\s*$", "", line)
                if line != original_line:
                    fixes += 1
            fixed_lines.append(line)

        return "\n".join(fixed_lines), fixes

    def _fix_structural_issues(self, content: str) -> Tuple[str, int]:
        """
        Fix structural markdown issues (MD022, MD031, MD032, MD029, MD036).

        - MD022: Headings surrounded by blank lines
        - MD031: Code blocks surrounded by blank lines
        - MD032: Lists surrounded by blank lines
        - MD029: Ordered list numbering
        - MD036: Bold text as headings
        """
        lines = content.split("\n")
        fixed_lines = []
        fixes = 0
        i = 0

        while i < len(lines):
            line = lines[i]
            prev_line = fixed_lines[-1] if fixed_lines else ""
            next_line = lines[i + 1] if i + 1 < len(lines) else ""

            # MD036: Convert **Bold:** to heading
            if re.match(r"^\*\*[^*]+\*\*:?\s*$", line.strip()):
                text = re.sub(r"\*\*([^*]+)\*\*:?", r"\1", line.strip())
                # Check if this looks like a heading (short, no lowercase middle)
                if len(text) < 50 and text[0].isupper():
                    fixes += 1
                    if prev_line.strip() != "":
                        fixed_lines.append("")
                    fixed_lines.append(f"### {text}")
                    if next_line.strip() != "":
                        fixed_lines.append("")
                    i += 1
                    continue

            # MD022: Headings should be surrounded by blank lines
            if re.match(r"^#{1,6}\s+", line):
                # Add blank line before heading (except at start)
                if fixed_lines and prev_line.strip() != "":
                    fixed_lines.append("")
                    fixes += 1

                fixed_lines.append(line)
                i += 1

                # Add blank line after heading
                if next_line.strip() != "" and not re.match(r"^#{1,6}\s+", next_line):
                    fixed_lines.append("")
                    fixes += 1
                continue

            # MD031: Code blocks should be surrounded by blank lines
            if line.strip().startswith("```"):
                # Add blank line before code block
                if fixed_lines and prev_line.strip() != "":
                    fixed_lines.append("")
                    fixes += 1

                # Add opening fence
                fixed_lines.append(line)
                i += 1

                # Copy code block content
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    fixed_lines.append(lines[i])
                    i += 1

                # Add closing fence
                if i < len(lines):
                    fixed_lines.append(lines[i])
                    i += 1

                # Add blank line after code block
                next_line = lines[i] if i < len(lines) else ""
                if next_line.strip() != "":
                    fixed_lines.append("")
                    fixes += 1
                continue

            # MD032: Lists should be surrounded by blank lines
            # MD029: Ordered lists should use 1. for all items
            if re.match(r"^\s*[-*+]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
                # Add blank line before list
                if fixed_lines and prev_line.strip() != "" and not self._is_list_item(prev_line):
                    fixed_lines.append("")
                    fixes += 1

                # Process list items
                list_indent = len(line) - len(line.lstrip())
                while i < len(lines):
                    curr_line = lines[i]

                    # Check if still in list
                    if not curr_line.strip():
                        # Empty line might be inside list
                        if i + 1 < len(lines) and self._is_list_item(lines[i + 1]):
                            fixed_lines.append(curr_line)
                            i += 1
                            continue
                        else:
                            break

                    # List item or continuation
                    if self._is_list_item(curr_line):
                        # MD029: Fix ordered list numbering
                        if re.match(r"^\s*\d+\.\s+", curr_line):
                            indent = len(curr_line) - len(curr_line.lstrip())
                            rest = re.sub(r"^\s*\d+\.", "", curr_line)
                            fixed_line = " " * indent + "1." + rest
                            if fixed_line != curr_line:
                                fixes += 1
                            fixed_lines.append(fixed_line)
                        else:
                            fixed_lines.append(curr_line)
                        i += 1
                    elif curr_line.startswith(" " * (list_indent + 2)):
                        # Continuation of list item (indented)
                        fixed_lines.append(curr_line)
                        i += 1
                    else:
                        break

                # Add blank line after list
                if i < len(lines) and lines[i].strip() != "":
                    fixed_lines.append("")
                    fixes += 1
                continue

            # Default: copy line as-is
            fixed_lines.append(line)
            i += 1

        return "\n".join(fixed_lines), fixes

    def _is_list_item(self, line: str) -> bool:
        """Check if line is a list item."""
        return bool(re.match(r"^\s*[-*+]\s+", line) or re.match(r"^\s*\d+\.\s+", line))

    def process_path(self, path: Path) -> Tuple[int, int]:
        """
        Process a file or directory.

        Args:
            path: Path to file or directory

        Returns:
            Tuple of (files_modified, total_fixes).
        """
        if not path.exists():
            print(f"Error: {path} does not exist", file=sys.stderr)
            return 0, 0

        files_to_fix = []
        if path.is_file():
            if path.suffix == ".md":
                files_to_fix.append(path)
            else:
                print(f"Warning: {path} is not a markdown file", file=sys.stderr)
                return 0, 0
        else:
            # Exclude certain directories
            exclude_patterns = {"node_modules", ".git", "venv", "__pycache__", ".tox"}
            files_to_fix = [f for f in path.rglob("*.md") if not any(part in exclude_patterns for part in f.parts)]

        if not files_to_fix:
            print(f"No markdown files found in {path}")
            return 0, 0

        print(f"Found {len(files_to_fix)} markdown file(s)")

        files_modified = 0
        total_fixes = 0

        for file_path in sorted(files_to_fix):
            modified, fixes = self.fix_file(file_path)
            if modified:
                files_modified += 1
                total_fixes += fixes

        return files_modified, total_fixes


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fix common markdown linting errors automatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("path", type=Path, help="Path to markdown file or directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Show what would be fixed without making changes")

    args = parser.parse_args()

    fixer = MarkdownFixer(verbose=args.verbose, dry_run=args.dry_run)
    files_modified, total_fixes = fixer.process_path(args.path)

    print("\nSummary:")
    print(f"  Files modified: {files_modified}")
    print(f"  Total fixes: {total_fixes}")

    if args.dry_run:
        print("\n[DRY RUN] No files were actually modified")

    return 0


if __name__ == "__main__":
    sys.exit(main())
