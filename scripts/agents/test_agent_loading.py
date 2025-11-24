#!/usr/bin/env python3
"""
Test agent discovery and loading.

This script tests that all agent configuration files can be discovered and parsed
correctly. It checks for duplicate names, readability issues, and basic structure.

Usage:
    python scripts/agents/test_agent_loading.py [--verbose]
    python scripts/agents/test_agent_loading.py --help

Exit Codes:
    0 - All agents loaded successfully
    1 - Errors found during loading
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import get_agents_dir

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


class AgentInfo:
    """Information about an agent configuration."""

    def __init__(self, file_path: Path, name: str, description: str, tools: str, model: str):
        self.file_path = file_path
        self.name = name
        self.description = description
        self.tools = tools
        self.model = model

    def __repr__(self):
        return f"AgentInfo(name={self.name}, file={self.file_path.name})"


def extract_frontmatter(content: str) -> Optional[str]:
    """
    Extract YAML frontmatter from markdown content.

    Args:
        content: The markdown file content

    Returns:
        Frontmatter text or None if not found
    """
    pattern = r'^---\s*\n(.*?\n)---\s*\n'
    match = re.match(pattern, content, re.DOTALL)
    return match.group(1) if match else None


def load_agent(file_path: Path, verbose: bool = False) -> Optional[AgentInfo]:
    """
    Load agent configuration from a markdown file.

    Args:
        file_path: Path to the agent markdown file
        verbose: Whether to show verbose output

    Returns:
        AgentInfo object or None if loading failed
    """
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"✗ {file_path.name}: Failed to read file - {e}", file=sys.stderr)
        return None

    # Extract frontmatter
    frontmatter_text = extract_frontmatter(content)
    if frontmatter_text is None:
        print(f"✗ {file_path.name}: No YAML frontmatter found", file=sys.stderr)
        return None

    # Parse YAML
    try:
        frontmatter = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as e:
        print(f"✗ {file_path.name}: YAML syntax error - {e}", file=sys.stderr)
        return None

    if not isinstance(frontmatter, dict):
        print(f"✗ {file_path.name}: Frontmatter is not a YAML mapping", file=sys.stderr)
        return None

    # Extract required fields
    name = frontmatter.get('name')
    description = frontmatter.get('description')
    tools = frontmatter.get('tools')
    model = frontmatter.get('model')

    # Validate required fields are present
    missing = []
    if not name:
        missing.append('name')
    if not description:
        missing.append('description')
    if not tools:
        missing.append('tools')
    if not model:
        missing.append('model')

    if missing:
        print(f"✗ {file_path.name}: Missing required fields: {', '.join(missing)}", file=sys.stderr)
        return None

    if verbose:
        print(f"✓ {file_path.name}: Loaded agent '{name}'")

    return AgentInfo(file_path, name, description, tools, model)


def check_for_duplicates(agents: List[AgentInfo]) -> List[Tuple[str, List[Path]]]:
    """
    Check for duplicate agent names.

    Args:
        agents: List of loaded agents

    Returns:
        List of (name, file_paths) for each duplicate name
    """
    name_to_files = {}
    for agent in agents:
        if agent.name not in name_to_files:
            name_to_files[agent.name] = []
        name_to_files[agent.name].append(agent.file_path)

    duplicates = [
        (name, files) for name, files in name_to_files.items()
        if len(files) > 1
    ]

    return duplicates


def test_agent_discovery(agents_dir: Path, verbose: bool = False) -> Tuple[List[AgentInfo], List[str]]:
    """
    Test discovery and loading of all agents.

    Args:
        agents_dir: Path to agents directory
        verbose: Whether to show verbose output

    Returns:
        Tuple of (loaded_agents, error_messages)
    """
    errors = []

    # Find all markdown files
    agent_files = sorted(agents_dir.glob('*.md'))

    if not agent_files:
        errors.append(f"No .md files found in {agents_dir}")
        return [], errors

    if verbose:
        print(f"Found {len(agent_files)} agent files\n")

    # Load each agent
    loaded_agents = []
    for file_path in agent_files:
        agent = load_agent(file_path, verbose)
        if agent:
            loaded_agents.append(agent)
        else:
            errors.append(f"Failed to load {file_path.name}")

    if verbose:
        print()

    # Check for duplicates
    duplicates = check_for_duplicates(loaded_agents)
    if duplicates:
        for name, files in duplicates:
            file_names = ', '.join(f.name for f in files)
            errors.append(f"Duplicate agent name '{name}' in files: {file_names}")

    return loaded_agents, errors


def display_agents(agents: List[AgentInfo]):
    """
    Display loaded agents in a formatted table.

    Args:
        agents: List of loaded agents
    """
    if not agents:
        print("No agents loaded")
        return

    # Calculate column widths
    max_name_len = max(len(agent.name) for agent in agents)
    max_file_len = max(len(agent.file_path.name) for agent in agents)

    # Print header
    print(f"\n{'Name':<{max_name_len}}  {'File':<{max_file_len}}  Model    Tools")
    print("-" * (max_name_len + max_file_len + 50))

    # Print each agent
    for agent in sorted(agents, key=lambda a: a.name):
        tools_list = agent.tools.split(',')[:3]  # First 3 tools
        tools_display = ','.join(t.strip() for t in tools_list)
        if len(agent.tools.split(',')) > 3:
            tools_display += ',...'

        print(f"{agent.name:<{max_name_len}}  "
              f"{agent.file_path.name:<{max_file_len}}  "
              f"{agent.model:<7}  "
              f"{tools_display}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test agent discovery and loading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test agent loading
    python scripts/agents/test_agent_loading.py

    # Test with verbose output
    python scripts/agents/test_agent_loading.py --verbose

    # Specify custom agents directory
    python scripts/agents/test_agent_loading.py --agents-dir /path/to/agents
        """
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output'
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

    # Find repository root
    repo_root = Path.cwd()
    while repo_root != repo_root.parent:
        if (repo_root / '.claude').exists():
            break
        repo_root = repo_root.parent
    else:
        print("Error: Could not find .claude directory", file=sys.stderr)
        return 1

    agents_dir = repo_root / args.agents_dir

    if not agents_dir.exists():
        print(f"Error: Agents directory not found: {agents_dir}", file=sys.stderr)
        return 1

    if not agents_dir.is_dir():
        print(f"Error: Not a directory: {agents_dir}", file=sys.stderr)
        return 1

    print(f"Testing agent discovery in {agents_dir.relative_to(repo_root)}/")

    # Test agent loading
    agents, errors = test_agent_discovery(agents_dir, verbose=args.verbose)

    # Display results
    if agents:
        display_agents(agents)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Loaded: {len(agents)} agents")

    if errors:
        print(f"Errors: {len(errors)}")
        print("\nError details:")
        for error in errors:
            print(f"  - {error}")
        return 1
    else:
        print("Status: ✓ All agents loaded successfully")
        return 0


if __name__ == '__main__':
    sys.exit(main())
