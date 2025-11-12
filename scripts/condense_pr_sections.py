#!/usr/bin/env python3
"""
Replace verbose PR Creation sections with references to CLAUDE.md.

The PR Creation section is identical across all 38 agent files (45 lines each).
Replace with a 5-line reference.
"""

import re
from pathlib import Path
from typing import Dict, Tuple

# Short reference for PR creation
PR_REFERENCE = """## Pull Request Creation

See [CLAUDE.md](../../CLAUDE.md#git-workflow) for complete PR creation instructions including linking to issues, verification steps, and requirements.

**Quick Summary**: Commit changes, push branch, create PR with `gh pr create --issue <issue-number>`, verify issue is linked."""


def replace_pr_section(content: str) -> Tuple[str, bool]:
    """Replace PR Creation section with reference."""
    # Pattern to match from "## Pull Request Creation" to the next "## "
    pattern = r'(## Pull Request Creation\n\n.*?)(^## )'

    match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
    if match:
        new_content = re.sub(pattern, PR_REFERENCE + '\n\n\\2', content, flags=re.MULTILINE | re.DOTALL)
        return new_content, True
    return content, False


def count_lines_removed(original: str, modified: str) -> int:
    """Count how many lines were removed."""
    return len(original.splitlines()) - len(modified.splitlines())


def process_file(filepath: Path) -> Dict:
    """Process a single agent file."""
    result = {
        'file': filepath.name,
        'pr_section_replaced': False,
        'lines_removed': 0
    }

    content = filepath.read_text()
    original_content = content

    content, changed = replace_pr_section(content)
    result['pr_section_replaced'] = changed

    if content != original_content:
        result['lines_removed'] = count_lines_removed(original_content, content)
        filepath.write_text(content)

    return result


def main():
    agents_dir = Path('/home/mvillmow/ml-odyssey-manual/.claude/agents')

    # Get all agent markdown files
    agent_files = sorted(agents_dir.glob('*.md'))

    results = []

    print("Condensing PR Creation sections...")
    print("=" * 80)

    for filepath in agent_files:
        result = process_file(filepath)
        results.append(result)

        if result['pr_section_replaced']:
            print(f"✓ {result['file']:50} | Lines removed: {result['lines_removed']:2}")
        else:
            print(f"○ {result['file']:50} | No PR section found")

    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Files processed: {len(results)}")
    print(f"  PR sections replaced: {sum(1 for r in results if r['pr_section_replaced'])}")
    print(f"  Total lines removed: {sum(r['lines_removed'] for r in results)}")


if __name__ == '__main__':
    main()
