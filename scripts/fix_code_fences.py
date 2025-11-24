#!/usr/bin/env python3
"""Fix duplicate code fences in markdown files."""

import re
from pathlib import Path


def fix_code_fences(filepath: Path):
    """Fix duplicate code fences in a markdown file."""
    content = filepath.read_text()

    # Fix duplicate code fences (```mojo followed immediately by another ```mojo)
    content = re.sub(r"```mojo\n```mojo\n", "```mojo\n", content)

    # Fix orphaned ```text that should close code blocks
    content = re.sub(r"\n```text\n\n", "\n```\n\n", content)

    filepath.write_text(content)
    print(f"Fixed {filepath}")


if __name__ == "__main__":
    api_ref = Path("docs/dev/api-reference.md")
    if api_ref.exists():
        fix_code_fences(api_ref)
