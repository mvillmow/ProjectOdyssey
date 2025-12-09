#!/usr/bin/env python3
"""
Validate Mojo syntax in documentation files.

This script detects deprecated Mojo syntax patterns in markdown documentation
files and reports them.

Usage:
    python scripts/validate_mojo_syntax_in_docs.py [directory]

Deprecated patterns detected:
- `inout self` in __init__ methods (should be `out self`)
- `inout self` in regular methods (should be `mut self`)
- `@value` decorator (should be `@fieldwise_init` + traits)
- `DynamicVector` (should be `List`)
- Tuple return syntax `-> (T1, T2)` (should be `-> Tuple[T1, T2]`)
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


class DeprecatedPattern:
    """Represents a deprecated Mojo syntax pattern."""

    def __init__(
        self, name: str, pattern: str, suggestion: str, context_lines: int = 2, in_code_blocks_only: bool = True
    ):
        self.name = name
        self.pattern = re.compile(pattern, re.MULTILINE)
        self.suggestion = suggestion
        self.context_lines = context_lines
        self.in_code_blocks_only = in_code_blocks_only


# Define deprecated patterns
DEPRECATED_PATTERNS = [
    DeprecatedPattern(
        name="inout self in __init__",
        pattern=r"fn\s+__init__\s*\(\s*inout\s+self",
        suggestion="Use `out self` instead of `inout self` in __init__ methods",
    ),
    DeprecatedPattern(
        name="inout self in regular methods",
        pattern=r"fn\s+(?!__init__|__moveinit__|__copyinit__|__del__)\w+\s*\(\s*inout\s+self",
        suggestion="Use `mut self` instead of `inout self` in regular methods",
    ),
    DeprecatedPattern(
        name="@value decorator",
        pattern=r"@value\s*\n\s*struct",
        suggestion="Use `@fieldwise_init` with traits instead of `@value`",
    ),
    DeprecatedPattern(
        name="DynamicVector usage", pattern=r"\bDynamicVector\b", suggestion="Use `List` instead of `DynamicVector`"
    ),
    DeprecatedPattern(
        name="Tuple return syntax",
        pattern=r"->\s*\([^)]+\)",
        suggestion="Use `-> Tuple[T1, T2]` instead of `-> (T1, T2)`",
    ),
]


def is_in_code_block(lines: List[str], line_num: int) -> bool:
    """Check if a line is inside a code block."""
    in_block = False
    for i, line in enumerate(lines[: line_num + 1]):
        if line.strip().startswith("```"):
            in_block = not in_block
    return in_block


def is_in_mojo_code_block(lines: List[str], line_num: int) -> bool:
    """Check if a line is inside a Mojo code block."""
    in_block = False
    block_lang = None

    for i, line in enumerate(lines[: line_num + 1]):
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_block:
                # Closing block
                in_block = False
                block_lang = None
            else:
                # Opening block
                in_block = True
                # Extract language identifier
                lang_match = re.match(r"```(\w+)", stripped)
                block_lang = lang_match.group(1) if lang_match else None

    return in_block and block_lang == "mojo"


def check_file(filepath: Path) -> List[Tuple[int, str, str, List[str]]]:
    """
    Check a file for deprecated patterns.

    Returns:
        List of (line_num, pattern_name, matched_text, context_lines)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        lines = content.splitlines()

    issues = []

    for pattern in DEPRECATED_PATTERNS:
        for match in pattern.pattern.finditer(content):
            line_num = content[: match.start()].count("\n")

            # Check if pattern should only be detected in code blocks
            if pattern.in_code_blocks_only:
                if not is_in_mojo_code_block(lines, line_num):
                    continue

            # Extract context lines
            start_line = max(0, line_num - pattern.context_lines)
            end_line = min(len(lines), line_num + pattern.context_lines + 1)
            context = lines[start_line:end_line]

            issues.append(
                (
                    line_num + 1,  # 1-indexed for display
                    pattern.name,
                    match.group(0),
                    context,
                )
            )

    return issues


def validate_directory(directory: Path, extensions: List[str] = [".md"]) -> int:
    """
    Validate all markdown files in a directory.

    Returns:
        Number of issues found
    """
    total_issues = 0
    files_checked = 0

    # Find all markdown files
    for ext in extensions:
        for filepath in directory.rglob(f"*{ext}"):
            # Skip hidden files and directories
            if any(part.startswith(".") for part in filepath.parts):
                continue

            files_checked += 1
            issues = check_file(filepath)

            if issues:
                print(f"\n{'=' * 80}")
                print(f"File: {filepath.relative_to(directory)}")
                print(f"{'=' * 80}")

                for line_num, pattern_name, matched_text, context in issues:
                    print(f"\nLine {line_num}: {pattern_name}")
                    print(f"  Matched: {matched_text!r}")
                    print(f"  Suggestion: {[p for p in DEPRECATED_PATTERNS if p.name == pattern_name][0].suggestion}")
                    print("\n  Context:")
                    for i, ctx_line in enumerate(context):
                        marker = ">>>" if i == len(context) // 2 else "   "
                        print(f"    {marker} {ctx_line}")

                    total_issues += 1

    print(f"\n{'=' * 80}")
    print(f"Summary: {total_issues} issues found in {files_checked} files")
    print(f"{'=' * 80}")

    return total_issues


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        directory = Path(sys.argv[1])
    else:
        directory = Path.cwd() / "shared"

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)

    print(f"Validating Mojo syntax in: {directory}")
    print(f"Checking for {len(DEPRECATED_PATTERNS)} deprecated patterns...")

    issue_count = validate_directory(directory)

    sys.exit(1 if issue_count > 0 else 0)


if __name__ == "__main__":
    main()
