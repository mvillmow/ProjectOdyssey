#!/usr/bin/env python3
"""
Check YAML frontmatter in agent configuration files.

This script verifies that all agent markdown files in .claude/agents/ have valid
YAML frontmatter with the required fields and correct types.

Usage:
    python scripts/agents/check_frontmatter.py [--verbose]
    python scripts/agents/check_frontmatter.py --help

Exit Codes:
    0 - All frontmatter is valid
    1 - Errors found in one or more files
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import get_agents_dir

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# Required fields and their expected types
REQUIRED_FIELDS = {
    'name': str,
    'description': str,
    'tools': str,
    'model': str,
}

# Optional fields and their expected types
OPTIONAL_FIELDS = {
    'level': int,
    'section': str,
    'workflow_phase': str,
}

# Valid model names
VALID_MODELS = {'sonnet', 'opus', 'haiku', 'claude-3-5-sonnet', 'claude-3-opus', 'claude-3-haiku'}


def extract_frontmatter(content: str) -> Optional[Tuple[str, int, int]]:
    """
    Extract YAML frontmatter from markdown content.

    Args:
        content: The markdown file content

    Returns:
        Tuple of (frontmatter_text, start_line, end_line) or None if no frontmatter found
    """
    # Match YAML frontmatter between --- delimiters
    pattern = r'^---\s*\n(.*?\n)---\s*\n'
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        return None

    frontmatter_text = match.group(1)
    start_line = 1
    end_line = content[:match.end()].count('\n')

    return frontmatter_text, start_line, end_line


def validate_field_type(field_name: str, value: Any, expected_type: type) -> Optional[str]:
    """
    Validate that a field has the expected type.

    Args:
        field_name: Name of the field
        value: The value to check
        expected_type: The expected Python type

    Returns:
        Error message if validation fails, None otherwise
    """
    if not isinstance(value, expected_type):
        return f"Field '{field_name}' should be {expected_type.__name__}, got {type(value).__name__}"
    return None


def validate_frontmatter(frontmatter: Dict, file_path: Path, verbose: bool = False) -> List[str]:
    """
    Validate the frontmatter data.

    Args:
        frontmatter: Parsed YAML frontmatter
        file_path: Path to the file being validated
        verbose: Whether to show verbose output

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check required fields
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in frontmatter:
            errors.append(f"Missing required field: '{field}'")
        else:
            error = validate_field_type(field, frontmatter[field], expected_type)
            if error:
                errors.append(error)

    # Check optional fields if present
    for field, expected_type in OPTIONAL_FIELDS.items():
        if field in frontmatter:
            error = validate_field_type(field, frontmatter[field], expected_type)
            if error:
                errors.append(error)

    # Validate model field value
    if 'model' in frontmatter:
        model = frontmatter['model']
        if model not in VALID_MODELS:
            errors.append(f"Invalid model '{model}'. Valid models: {', '.join(sorted(VALID_MODELS))}")

    # Validate name field format (should be lowercase with hyphens)
    if 'name' in frontmatter:
        name = frontmatter['name']
        if not re.match(r'^[a-z][a-z0-9-]*$', name):
            errors.append(f"Name '{name}' should be lowercase with hyphens (e.g., 'chief-architect')")

    # Validate tools field (should be comma-separated list)
    if 'tools' in frontmatter:
        tools = frontmatter['tools']
        if not tools.strip():
            errors.append("Tools field cannot be empty")

    # Check for unexpected fields
    all_valid_fields = set(REQUIRED_FIELDS.keys()) | set(OPTIONAL_FIELDS.keys())
    for field in frontmatter.keys():
        if field not in all_valid_fields:
            if verbose:
                errors.append(f"Warning: Unexpected field '{field}'")

    return errors


def check_file(file_path: Path, verbose: bool = False) -> Tuple[bool, List[str]]:
    """
    Check a single agent configuration file.

    Args:
        file_path: Path to the markdown file
        verbose: Whether to show verbose output

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        return False, [f"Failed to read file: {e}"]

    # Extract frontmatter
    result = extract_frontmatter(content)
    if result is None:
        return False, ["No YAML frontmatter found (should start with --- and end with ---)"]

    frontmatter_text, start_line, end_line = result

    # Parse YAML
    try:
        frontmatter = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as e:
        return False, [f"YAML syntax error: {e}"]

    if frontmatter is None:
        return False, ["Empty frontmatter"]

    if not isinstance(frontmatter, dict):
        return False, [f"Frontmatter should be a YAML mapping, got {type(frontmatter).__name__}"]

    # Validate frontmatter content
    errors = validate_frontmatter(frontmatter, file_path, verbose)

    return len(errors) == 0, errors


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check YAML frontmatter in agent configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check all agent files
    python scripts/agents/check_frontmatter.py

    # Check with verbose output
    python scripts/agents/check_frontmatter.py --verbose
        """
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output including warnings'
    )
    parser.add_argument(
        '--agents-dir',
        type=Path,
        default=None,  # Will use get_agents_dir() if not specified
        help='Path to agents directory (default: .claude/agents)'
    )

    args = parser.parse_args()
    # Use get_agents_dir() if no custom path specified
    if args.agents_dir is None:
        args.agents_dir = get_agents_dir()

    # Find repository root (directory containing .claude)
    repo_root = Path.cwd()
    while repo_root != repo_root.parent:
        if (repo_root / '.claude').exists():
            break
        repo_root = repo_root.parent
    else:
        print("Error: Could not find .claude directory in current or parent directories", file=sys.stderr)
        return 1

    agents_dir = repo_root / args.agents_dir

    if not agents_dir.exists():
        print(f"Error: Agents directory not found: {agents_dir}", file=sys.stderr)
        return 1

    if not agents_dir.is_dir():
        print(f"Error: Not a directory: {agents_dir}", file=sys.stderr)
        return 1

    # Find all markdown files
    agent_files = sorted(agents_dir.glob('*.md'))

    if not agent_files:
        print(f"Error: No .md files found in {agents_dir}", file=sys.stderr)
        return 1

    print(f"Checking {len(agent_files)} agent configuration files in {agents_dir.relative_to(repo_root)}/\n")

    # Check each file
    all_valid = True
    for file_path in agent_files:
        is_valid, errors = check_file(file_path, verbose=args.verbose)

        if is_valid:
            if args.verbose:
                print(f"✓ {file_path.name}")
        else:
            all_valid = False
            print(f"✗ {file_path.name}")
            for error in errors:
                print(f"  - {error}")
            print()

    # Summary
    print("-" * 60)
    if all_valid:
        print(f"✓ All {len(agent_files)} agent files have valid frontmatter")
        return 0
    else:
        failed_count = sum(1 for f in agent_files if not check_file(f, verbose=args.verbose)[0])
        print(f"✗ {failed_count} file(s) have errors")
        return 1


if __name__ == '__main__':
    sys.exit(main())
