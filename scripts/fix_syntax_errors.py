#!/usr/bin/env python3
"""Fix syntax errors introduced by overly aggressive period additions.

The section_body_endings fix incorrectly added periods to code statements
that appeared after docstrings. This script removes those periods.
"""

import re
from pathlib import Path


def fix_trailing_periods_in_code(content: str) -> tuple[str, int]:
    """Remove trailing periods from code statements (not docstrings)."""
    fixes = 0

    # Pattern 1: Lines ending with : followed by period (multiline function signatures)
    # ) -> Type:. -> ) -> Type:
    pattern1 = r"^(.+):\.$"
    matches1 = len(re.findall(pattern1, content, flags=re.MULTILINE))
    content = re.sub(pattern1, r"\1:", content, flags=re.MULTILINE)
    fixes += matches1

    # Pattern 2: Import statements with period
    # from .module import x. -> from .module import x
    pattern2 = r"^(\s+from .+ import .+)\.$"
    matches2 = len(re.findall(pattern2, content, flags=re.MULTILINE))
    content = re.sub(pattern2, r"\1", content, flags=re.MULTILINE)
    fixes += matches2

    # Pattern 3: Any assignment statement ending with period (must not be in a string)
    # ptr[i] = val. -> ptr[i] = val
    # self.field = value. -> self.field = value
    pattern3 = r"^(\s+.+=.+[^.\"\s])\.$"
    matches3 = len(re.findall(pattern3, content, flags=re.MULTILINE))
    content = re.sub(pattern3, r"\1", content, flags=re.MULTILINE)
    fixes += matches3

    # Pattern 4: String concatenation expressions ending with period
    # return "text" + var + ")". -> return "text" + var + ")"
    pattern4 = r"^(\s+.*\".+)\"\.$"
    matches4 = len(re.findall(pattern4, content, flags=re.MULTILINE))
    content = re.sub(pattern4, r'\1"', content, flags=re.MULTILINE)
    fixes += matches4

    # Pattern 5: Any line with code ending in period (not comment, not string literal)
    # Catches most remaining cases
    pattern5 = r"^(\s+[^#\"'\n]+[^.\s])\.$"
    matches5 = len(re.findall(pattern5, content, flags=re.MULTILINE))
    content = re.sub(pattern5, r"\1", content, flags=re.MULTILINE)
    fixes += matches5

    return content, fixes


def main():
    base_path = Path("/home/mvillmow/ml-odyssey")
    files = list((base_path / "shared").rglob("*.mojo"))

    total_fixes = 0

    for file_path in files:
        content = file_path.read_text()
        fixed_content, fixes = fix_trailing_periods_in_code(content)

        if fixes > 0:
            file_path.write_text(fixed_content)
            print(f"Fixed {fixes} syntax errors in {file_path.relative_to(base_path)}")
            total_fixes += fixes

    print(f"\nTotal: {total_fixes} syntax errors fixed")


if __name__ == "__main__":
    main()
