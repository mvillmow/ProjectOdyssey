#!/usr/bin/env python3

"""
Fix remaining markdown linting errors identified by pre-commit.

This script fixes:
- MD040: Add missing language tags to code blocks
- MD026: Remove trailing punctuation from headings
- MD036: Convert bold headings to actual headings
- MD029: Fix ordered list numbering
"""

import re
from pathlib import Path

def fix_file(file_path: Path) -> bool:
    """Fix markdown issues in a file."""
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception:
        return False

    original = content

    # Fix MD040: Add language tags to code blocks
    # Find ``` without a language tag
    content = re.sub(
        r'^```\n',  # ``` followed by newline (no language)
        '```text\n',  # Add 'text' language tag
        content,
        flags=re.MULTILINE
    )

    # Fix MD026: Remove trailing colons from headings
    content = re.sub(
        r'^(#{1,6}\s+[^:]+):\s*$',
        r'\1',
        content,
        flags=re.MULTILINE
    )

    # Fix MD036: Convert **Bold** used as headings to actual headings
    # This is tricky - only convert if it looks like a section header
    # (at start of line, followed by colon or newline)
    lines = content.split('\n')
    fixed_lines = []
    for i, line in enumerate(lines):
        # Check if line is just bold text (potential heading)
        if re.match(r'^\*\*[^*]+\*\*:\s*$', line):
            # Extract the bold text
            text = re.sub(r'\*\*([^*]+)\*\*:', r'\1', line)
            # Convert to heading (### level)
            fixed_lines.append(f'### {text}')
        else:
            fixed_lines.append(line)

    content = '\n'.join(fixed_lines)

    # Fix MD029: Ensure ordered lists use 1. prefix
    # This requires careful handling
    lines = content.split('\n')
    fixed_lines = []
    in_list = False
    list_start = 0

    for i, line in enumerate(lines):
        # Check if line is an ordered list item
        if re.match(r'^\s*\d+\.\s+', line):
            if not in_list:
                in_list = True
                list_start = i
            # Replace numbering with 1.
            fixed_line = re.sub(r'^(\s*)\d+\.(\s+)', r'\g<1>1.\g<2>', line)
            fixed_lines.append(fixed_line)
        elif in_list and (line.strip() == '' or not re.match(r'^\s+', line)):
            in_list = False
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    content = '\n'.join(fixed_lines)

    if content != original:
        file_path.write_text(content, encoding='utf-8')
        return True

    return False

def main():
    """Fix remaining markdown errors."""
    repo_root = Path('/home/mvillmow/ml-odyssey-manual')

    # Files with known issues
    problem_files = [
        '.claude/agents/architecture-design.md',
        '.claude/agents/agentic-workflows-orchestrator.md',
        '.claude/agents/chief-architect.md',
        '.claude/agents/dependency-review-specialist.md',
        '.claude/agents/documentation-engineer.md',
        '.claude/agents/documentation-review-specialist.md',
        '.claude/agents/documentation-specialist.md',
        '.claude/agents/foundation-orchestrator.md',
        '.claude/agents/junior-documentation-engineer.md',
        '.claude/agents/senior-implementation-engineer.md',
        '.claude/agents/test-engineer.md',
        'agents/templates/level-1-section-orchestrator.md',
        'agents/templates/level-2-module-design.md',
        'agents/templates/level-3-component-specialist.md',
        'agents/templates/level-5-junior-engineer.md',
        'notes/issues/64/README.md',
    ]

    fixed_count = 0
    for file_rel in problem_files:
        file_path = repo_root / file_rel
        if file_path.exists() and fix_file(file_path):
            fixed_count += 1
            print(f"Fixed: {file_rel}")

    print(f"\nTotal files fixed: {fixed_count}")

if __name__ == '__main__':
    main()
