#!/usr/bin/env python3
"""
Fix markdown linting errors in agent files.

This script systematically fixes common markdown linting issues:
1. Removes PR template sections (## Changes, ## Testing, footer text)
2. Fixes multiple consecutive blank lines (2+ → 1)
3. Adds blank lines around headings, lists, code blocks
4. Fixes line length > 120 characters
5. Fixes code blocks without language tags
6. Fixes inline HTML like <issue-number>

Usage:
    python3 scripts/fix_agent_markdown.py
"""

import re
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common import get_agents_dir


def remove_pr_template_sections(content: str) -> str:
    """Remove PR template sections and footer text."""
    # Remove ## Changes section
    content = re.sub(
        r'\n## Changes\n.*?(?=\n##|\Z)',
        '',
        content,
        flags=re.DOTALL
    )

    # Remove ## Testing section
    content = re.sub(
        r'\n## Testing\n.*?(?=\n##|\Z)',
        '',
        content,
        flags=re.DOTALL
    )

    # Remove footer text (Generated with Claude Code, Co-Authored-By)
    content = re.sub(
        r'\n---\n\n.*?Generated with.*?Co-Authored-By:.*?\n',
        '',
        content,
        flags=re.DOTALL
    )

    # Remove standalone footer markers
    content = re.sub(
        r'\n---\n\nGenerated with.*?Co-Authored-By:.*?\n',
        '',
        content,
        flags=re.DOTALL
    )

    return content


def fix_multiple_blank_lines(content: str) -> str:
    """Replace multiple consecutive blank lines with single blank line."""
    # Replace 2+ blank lines with 1 blank line
    content = re.sub(r'\n\n\n+', '\n\n', content)
    return content


def add_blank_lines_around_headings(content: str) -> str:
    """Ensure headings have blank lines before and after."""
    lines = content.split('\n')
    result = []

    for i, line in enumerate(lines):
        # Check if current line is a heading
        if re.match(r'^#{1,6}\s+', line):
            # Add blank line before heading if needed
            if i > 0 and result and result[-1].strip() != '':
                result.append('')

            result.append(line)

            # Add blank line after heading if needed
            if i < len(lines) - 1 and lines[i + 1].strip() != '':
                result.append('')
        else:
            result.append(line)

    return '\n'.join(result)


def add_blank_lines_around_code_blocks(content: str) -> str:
    """Ensure code blocks have blank lines before and after."""
    lines = content.split('\n')
    result = []
    in_code_block = False

    for i, line in enumerate(lines):
        # Check if line starts or ends a code block
        if line.strip().startswith('```'):
            if not in_code_block:
                # Starting code block - add blank line before if needed
                if i > 0 and result and result[-1].strip() != '':
                    result.append('')
                in_code_block = True
            else:
                # Ending code block
                in_code_block = False
                result.append(line)
                # Add blank line after if needed
                if i < len(lines) - 1 and lines[i + 1].strip() != '':
                    result.append('')
                continue

        result.append(line)

    return '\n'.join(result)


def add_blank_lines_around_lists(content: str) -> str:
    """Ensure lists have blank lines before and after."""
    lines = content.split('\n')
    result = []
    in_list = False

    for i, line in enumerate(lines):
        # Check if line is a list item
        is_list_item = bool(re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+\.\s+', line))

        if is_list_item and not in_list:
            # Starting a list - add blank line before if needed
            if i > 0 and result and result[-1].strip() != '':
                result.append('')
            in_list = True
        elif not is_list_item and in_list:
            # Ending a list - add blank line after previous item
            if line.strip() != '':
                result.append('')
            in_list = False

        result.append(line)

    return '\n'.join(result)


def fix_code_block_language_tags(content: str) -> str:
    """Add language tags to code blocks without them."""
    # Find code blocks without language tags
    # Pattern: ``` followed by newline (not a language identifier)
    content = re.sub(r'```\n', '```text\n', content)
    return content


def fix_inline_html(content: str) -> str:
    """Replace inline HTML with markdown equivalents."""
    # Replace <issue-number> with `issue-number`
    content = re.sub(r'<([^>]+)>', r'`\1`', content)
    return content


def fix_long_lines(content: str) -> str:
    """Break lines longer than 120 characters."""
    lines = content.split('\n')
    result = []
    in_code_block = False

    for line in lines:
        # Track code blocks
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            result.append(line)
            continue

        # Skip code blocks and lines with URLs
        if in_code_block or 'http://' in line or 'https://' in line:
            result.append(line)
            continue

        # Skip list items, headings, and already short lines
        if len(line) <= 120 or re.match(r'^\s*[-*+]\s+', line) or re.match(r'^#{1,6}\s+', line):
            result.append(line)
            continue

        # Break long lines at sentence boundaries
        if len(line) > 120:
            # Try to break at period, comma, or space around character 100-120
            words = line.split()
            current_line = []
            current_length = 0

            for word in words:
                if current_length + len(word) + 1 > 120 and current_line:
                    result.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    current_line.append(word)
                    current_length += len(word) + 1

            if current_line:
                result.append(' '.join(current_line))
        else:
            result.append(line)

    return '\n'.join(result)


def ensure_file_ends_with_newline(content: str) -> str:
    """Ensure file ends with exactly one newline."""
    content = content.rstrip('\n')
    return content + '\n'


def fix_markdown_file(file_path: Path) -> bool:
    """Fix all markdown linting issues in a file."""
    print(f"Processing: {file_path.name}")

    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content

        # Apply all fixes in order
        content = remove_pr_template_sections(content)
        content = fix_inline_html(content)
        content = fix_code_block_language_tags(content)
        content = add_blank_lines_around_code_blocks(content)
        content = add_blank_lines_around_headings(content)
        content = add_blank_lines_around_lists(content)
        content = fix_long_lines(content)
        content = fix_multiple_blank_lines(content)
        content = ensure_file_ends_with_newline(content)

        # Write back if changed
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            print(f"  ✓ Fixed {file_path.name}")
            return True
        else:
            print(f"  - No changes needed for {file_path.name}")
            return False

    except Exception as e:
        print(f"  ✗ Error processing {file_path.name}: {e}")
        return False


def main() -> None:
    """Main function to fix all agent markdown files."""
    agents_dir = get_agents_dir()

    if not agents_dir.exists():
        print(f"Error: Directory not found: {agents_dir}")
        return 1

    # Find all markdown files
    md_files = sorted(agents_dir.glob('*.md'))

    if not md_files:
        print(f"No markdown files found in {agents_dir}")
        return 1

    print(f"Found {len(md_files)} markdown files to process\n")

    fixed_count = 0
    for md_file in md_files:
        if fix_markdown_file(md_file):
            fixed_count += 1

    print(f"\n{'='*60}")
    print(f"Summary: Fixed {fixed_count}/{len(md_files)} files")
    print(f"{'='*60}")

    return 0


if __name__ == '__main__':
    exit(main())
