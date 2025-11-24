#!/usr/bin/env python3
"""
Fix duplicate Delegation sections created by our script.

Some files had an existing "## Delegation" section with subsections,
and we added another "## Delegation" section with the reference.
This script merges them properly.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common import get_agents_dir

def fix_duplicate_delegation(content: str) -> Tuple[str, bool]:
    """
    Fix duplicate Delegation sections.

    Pattern to find:
    ## Delegation

    ### Delegates To
    ...

    ### Coordinates With
    ...

    ## Delegation

    For standard delegation patterns...

    Replace with:
    ## Delegation

    ### Delegates To
    ...

    ### Coordinates With
    ...

    ### Skip-Level Guidelines

    For standard delegation patterns...
    """

    # Pattern to match duplicate Delegation sections
    pattern = r'(## Delegation\n\n(?:### Delegates To\n.*?)?(?:### Coordinates With\n.*?)?)(\n## Delegation\n\n)(For standard delegation patterns.*?)(\n\n## )'

    match = re.search(pattern, content, re.DOTALL)
    if match:
        # Reconstruct with proper structure
        first_part = match.group(1)  # First Delegation section with subsections
        reference = match.group(3)   # The reference text
        next_section = match.group(4) # Next section header

        # Create merged section
        merged = f"{first_part}\n\n### Skip-Level Guidelines\n\n{reference}{next_section}"

        # Replace in content
        new_content = content[:match.start()] + merged + content[match.end():]
        return new_content, True

    return content, False


def process_file(filepath: Path) -> Dict:
    """Process a single agent file."""
    result = {
        'file': filepath.name,
        'fixed': False,
    }

    content = filepath.read_text()
    original_content = content

    new_content, changed = fix_duplicate_delegation(content)
    result['fixed'] = changed

    if changed:
        filepath.write_text(new_content)

    return result


def main() -> None:
    agents_dir = get_agents_dir()

    # Get all agent markdown files
    agent_files = sorted(agents_dir.glob('*.md'))

    results = []

    print("Fixing duplicate Delegation sections...")
    print("=" * 80)

    for filepath in agent_files:
        result = process_file(filepath)
        results.append(result)

        if result['fixed']:
            print(f"✓ {result['file']:50} | Fixed")

    print("=" * 80)
    print("\nSummary:")
    print(f"  Files processed: {len(results)}")
    print(f"  Files fixed: {sum(1 for r in results if r['fixed'])}")


if __name__ == '__main__':
    main()
