#!/usr/bin/env python3
"""
List all available agents.

This script displays all agent configurations organized by level (0-5),
showing their name, description, and available tools.

Usage:
    python scripts/agents/list_agents.py [--level LEVEL] [--verbose]
    python scripts/agents/list_agents.py --help

Exit Codes:
    0 - Success
    1 - Errors occurred
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import get_agents_dir

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


class AgentInfo:
    """Information about an agent configuration."""

    def __init__(self, file_path: Path, frontmatter: Dict):
        self.file_path = file_path
        self.name = frontmatter.get("name", "unknown")
        self.description = frontmatter.get("description", "No description")
        self.tools = frontmatter.get("tools", "")
        self.model = frontmatter.get("model", "unknown")
        self.level = self._infer_level(frontmatter)

    def _infer_level(self, frontmatter: Dict) -> int:
        """
        Infer agent level from frontmatter or name.

        Level hierarchy:
        - 0: Meta-orchestrator (Chief Architect)
        - 1: Section Orchestrators (foundation, shared-library, tooling, papers, cicd, agentic-workflows)
        - 2: Design Agents (architecture, integration, security)
        - 3: Component Specialists (implementation, test, documentation, performance, security)
        - 4: Senior Engineers
        - 5: Junior Engineers
        """
        # Check if level is explicitly specified
        if "level" in frontmatter:
            return frontmatter["level"]

        # Infer from name
        name = self.name.lower()

        if "chief-architect" in name:
            return 0
        elif "orchestrator" in name:
            return 1
        elif "design" in name:
            return 2
        elif "specialist" in name:
            return 3
        elif "senior" in name:
            return 4
        elif "junior" in name:
            return 5
        elif "engineer" in name:
            # Default engineers to level 4 unless junior
            return 4
        else:
            # Unknown - default to middle level
            return 3

    def get_tools_list(self) -> List[str]:
        """Get list of tool names."""
        if not self.tools:
            return []
        return [t.strip() for t in self.tools.split(",")]

    def __repr__(self):
        return f"AgentInfo(level={self.level}, name={self.name})"


def extract_frontmatter(content: str) -> Optional[str]:
    """
    Extract YAML frontmatter from markdown content.

    Args:
        content: The markdown file content

    Returns:
        Frontmatter text or None if not found
    """
    pattern = r"^---\s*\n(.*?\n)---\s*\n"
    match = re.match(pattern, content, re.DOTALL)
    return match.group(1) if match else None


def load_agent(file_path: Path) -> Optional[AgentInfo]:
    """
    Load agent configuration from a markdown file.

    Args:
        file_path: Path to the agent markdown file

    Returns:
        AgentInfo object or None if loading failed
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Warning: Failed to read {file_path.name}: {e}", file=sys.stderr)
        return None

    frontmatter_text = extract_frontmatter(content)
    if frontmatter_text is None:
        print(f"Warning: No frontmatter in {file_path.name}", file=sys.stderr)
        return None

    try:
        frontmatter = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as e:
        print(f"Warning: Invalid YAML in {file_path.name}: {e}", file=sys.stderr)
        return None

    if not isinstance(frontmatter, dict):
        print(f"Warning: Frontmatter is not a mapping in {file_path.name}", file=sys.stderr)
        return None

    return AgentInfo(file_path, frontmatter)


def load_all_agents(agents_dir: Path) -> List[AgentInfo]:
    """
    Load all agent configurations from a directory.

    Args:
        agents_dir: Path to agents directory

    Returns:
        List of AgentInfo objects
    """
    agent_files = sorted(agents_dir.glob("*.md"))
    agents = []

    for file_path in agent_files:
        agent = load_agent(file_path)
        if agent:
            agents.append(agent)

    return agents


def group_by_level(agents: List[AgentInfo]) -> Dict[int, List[AgentInfo]]:
    """
    Group agents by their level.

    Args:
        agents: List of agents

    Returns:
        Dictionary mapping level to list of agents
    """
    grouped = {}
    for agent in agents:
        if agent.level not in grouped:
            grouped[agent.level] = []
        grouped[agent.level].append(agent)

    # Sort agents within each level by name
    for level in grouped:
        grouped[level].sort(key=lambda a: a.name)

    return grouped


def format_description(description: str, max_width: int = 60, indent: int = 0) -> str:
    """
    Format description text with word wrapping.

    Args:
        description: Description text
        max_width: Maximum line width
        indent: Indentation for wrapped lines

    Returns:
        Formatted description
    """
    words = description.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(" ".join(current_line))

    indent_str = " " * indent
    return ("\n" + indent_str).join(lines)


def display_agents(agents: List[AgentInfo], verbose: bool = False, level_filter: Optional[int] = None):
    """
    Display agents organized by level.

    Args:
        agents: List of agents to display
        verbose: Whether to show detailed information
        level_filter: If specified, only show agents at this level
    """
    level_names = {
        0: "Level 0: Meta-Orchestrator",
        1: "Level 1: Section Orchestrators",
        2: "Level 2: Design Agents",
        3: "Level 3: Component Specialists",
        4: "Level 4: Senior Engineers",
        5: "Level 5: Junior Engineers",
    }

    grouped = group_by_level(agents)

    # Filter by level if specified
    if level_filter is not None:
        grouped = {level_filter: grouped.get(level_filter, [])}

    if not grouped:
        print("No agents found")
        return

    total_agents = sum(len(agents_list) for agents_list in grouped.values())
    print(f"\nTotal Agents: {total_agents}\n")

    for level in sorted(grouped.keys()):
        agents_list = grouped[level]
        level_name = level_names.get(level, f"Level {level}")

        print(f"\n{'=' * 70}")
        print(f"{level_name} ({len(agents_list)} agents)")
        print("=" * 70)

        for agent in agents_list:
            print(f"\n{agent.name}")
            print("-" * len(agent.name))

            if verbose:
                # Verbose mode: show all details
                print(f"File:        {agent.file_path.name}")
                print(f"Model:       {agent.model}")
                print(f"Description: {format_description(agent.description, max_width=55, indent=13)}")
                print(f"Tools:       {', '.join(agent.get_tools_list())}")
            else:
                # Compact mode: description and first few tools
                print(f"{format_description(agent.description, max_width=70)}")
                tools = agent.get_tools_list()
                if tools:
                    tools_display = ", ".join(tools[:5])
                    if len(tools) > 5:
                        tools_display += f", ... ({len(tools)} total)"
                    print(f"Tools: {tools_display}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="List all available agents organized by level",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Agent Levels:
    0 - Meta-Orchestrator (Chief Architect)
    1 - Section Orchestrators (Foundation, Shared Library, Tooling, Papers, CI/CD, Agentic Workflows)
    2 - Design Agents (Architecture, Integration, Security)
    3 - Component Specialists (Implementation, Test, Documentation, Performance, Security)
    4 - Senior Engineers
    5 - Junior Engineers

Examples:
    # List all agents
    python scripts/agents/list_agents.py

    # List agents with verbose details
    python scripts/agents/list_agents.py --verbose

    # List only level 1 agents (Section Orchestrators)
    python scripts/agents/list_agents.py --level 1

    # List only level 5 agents (Junior Engineers)
    python scripts/agents/list_agents.py --level 5 --verbose
        """,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information for each agent")
    parser.add_argument(
        "--level", "-l", type=int, choices=[0, 1, 2, 3, 4, 5], help="Show only agents at this level (0-5)"
    )
    parser.add_argument(
        "--agents-dir",
        type=Path,
        default=None,  # Will use get_agents_dir() if not specified
        help="Path to agents directory (default: .claude/agents)",
    )

    args = parser.parse_args()
    # Use get_agents_dir() if no custom path specified
    if args.agents_dir is None:
        args.agents_dir = get_agents_dir()

    # Find repository root
    repo_root = Path.cwd()
    while repo_root != repo_root.parent:
        if (repo_root / ".claude").exists():
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

    # Load all agents
    agents = load_all_agents(agents_dir)

    if not agents:
        print(f"Error: No agents loaded from {agents_dir}", file=sys.stderr)
        return 1

    # Display agents
    display_agents(agents, verbose=args.verbose, level_filter=args.level)

    return 0


if __name__ == "__main__":
    sys.exit(main())
