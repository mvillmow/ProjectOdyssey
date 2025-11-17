#!/usr/bin/env python3
"""
Remove redundant sections from agent files.

This script:
1. Removes Skip-Level Delegation sections and replaces with references
2. Removes Error Handling & Recovery sections from orchestrators
3. Tracks changes made
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

# Import shared utilities
from common import get_agents_dir

# Define the replacement text
SKIP_LEVEL_REPLACEMENT = """## Delegation

For standard delegation patterns, escalation rules, and skip-level guidelines, see [delegation-rules.md](../delegation-rules.md#skip-level-delegation).

**Quick Summary**: Follow hierarchy for all non-trivial work. Skip-level delegation is acceptable only for truly trivial fixes (< 20 lines, no design decisions)."""

ERROR_HANDLING_REPLACEMENT = """## Error Handling

For comprehensive error handling, recovery strategies, and escalation protocols, see [orchestration-patterns.md](../../notes/review/orchestration-patterns.md#error-handling--recovery).

**Quick Summary**: Classify errors (transient/permanent/blocker), retry transient errors up to 3 times, escalate blockers with detailed report."""


def remove_skip_level_delegation(content: str) -> Tuple[str, bool]:
    """Remove Skip-Level Delegation section and replace with reference."""
    # Pattern to match the entire Skip-Level Delegation section
    pattern = r'## Skip-Level Delegation\n\n.*?(?=\n## )'

    match = re.search(pattern, content, re.DOTALL)
    if match:
        new_content = re.sub(pattern, SKIP_LEVEL_REPLACEMENT + '\n\n', content, flags=re.DOTALL)
        return new_content, True
    return content, False


def remove_error_handling(content: str) -> Tuple[str, bool]:
    """Remove Error Handling & Recovery section and replace with reference."""
    # Pattern to match the entire Error Handling section
    pattern = r'## Error Handling & Recovery\n\n.*?(?=\n## )'

    match = re.search(pattern, content, re.DOTALL)
    if match:
        new_content = re.sub(pattern, ERROR_HANDLING_REPLACEMENT + '\n\n', content, flags=re.DOTALL)
        return new_content, True
    return content, False


def count_lines_removed(original: str, modified: str) -> int:
    """Count how many lines were removed."""
    return len(original.splitlines()) - len(modified.splitlines())


def process_file(filepath: Path, process_skip_level: bool = True, process_error_handling: bool = False) -> Dict:
    """Process a single agent file."""
    result = {
        'file': filepath.name,
        'skip_level_removed': False,
        'error_handling_removed': False,
        'lines_removed': 0
    }

    content = filepath.read_text()
    original_content = content

    if process_skip_level:
        content, changed = remove_skip_level_delegation(content)
        result['skip_level_removed'] = changed

    if process_error_handling:
        content, changed = remove_error_handling(content)
        result['error_handling_removed'] = changed

    if content != original_content:
        result['lines_removed'] = count_lines_removed(original_content, content)
        filepath.write_text(content)

    return result


def main():
    agents_dir = get_agents_dir()

    # Files with Skip-Level Delegation sections
    skip_level_files = [
        'cicd-orchestrator.md',
        'chief-architect.md',
        'implementation-specialist.md',
        'integration-design.md',
        'architecture-design.md',
        'tooling-orchestrator.md',
        'test-specialist.md',
        'papers-orchestrator.md',
        'performance-specialist.md',
        'security-design.md',
        'security-specialist.md',
        'shared-library-orchestrator.md',
        'documentation-specialist.md',
        # foundation-orchestrator.md already processed
        'agentic-workflows-orchestrator.md',
    ]

    # Orchestrators that should have Error Handling sections removed
    orchestrator_files = [
        'cicd-orchestrator.md',
        # 'foundation-orchestrator.md', # already processed
        'tooling-orchestrator.md',
        'papers-orchestrator.md',
        'shared-library-orchestrator.md',
        'agentic-workflows-orchestrator.md',
    ]

    results = []

    print("Processing agent files...")
    print("=" * 80)

    # Process files with skip-level delegation
    for filename in skip_level_files:
        filepath = agents_dir / filename
        if filepath.exists():
            is_orchestrator = filename in orchestrator_files
            result = process_file(filepath, process_skip_level=True, process_error_handling=is_orchestrator)
            results.append(result)

            status = []
            if result['skip_level_removed']:
                status.append('Skip-Level')
            if result['error_handling_removed']:
                status.append('Error Handling')

            if status:
                print(f"✓ {filename:45} | Removed: {', '.join(status):30} | Lines: {result['lines_removed']:3}")
            else:
                print(f"○ {filename:45} | No changes needed")

    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Files processed: {len(results)}")
    print(f"  Skip-Level sections removed: {sum(1 for r in results if r['skip_level_removed'])}")
    print(f"  Error Handling sections removed: {sum(1 for r in results if r['error_handling_removed'])}")
    print(f"  Total lines removed: {sum(r['lines_removed'] for r in results)}")


if __name__ == '__main__':
    main()
