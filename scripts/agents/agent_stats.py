#!/usr/bin/env python3
"""
agent_stats.py - Generate usage statistics for the agent system

This script analyzes the agent configuration files and generates comprehensive
statistics including:
- Agent counts by level
- Tool usage across agents
- Skill references
- Delegation patterns
- Cross-references between agents

Usage:
    python3 scripts/agents/agent_stats.py [options]

Options:
    --format {text,json,markdown}  Output format (default: text)
    --output FILE                   Write output to file (default: stdout)
    --verbose                       Include detailed breakdowns
    --help                          Show this help message

Examples:
    # Generate text report to stdout
    python3 scripts/agents/agent_stats.py

    # Generate markdown report to file
    python3 scripts/agents/agent_stats.py --format markdown --output stats.md

    # Generate JSON with verbose details
    python3 scripts/agents/agent_stats.py --format json --verbose
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


class AgentAnalyzer:
    """Analyze agent configuration files and generate statistics."""

    def __init__(self, agents_dir: Path, verbose: bool = False):
        """Initialize the analyzer.

        Args:
            agents_dir: Path to .claude/agents directory
            verbose: Whether to include detailed breakdowns
        """
        self.agents_dir = agents_dir
        self.verbose = verbose
        self.agents: List[Dict] = []
        self.stats: Dict = {}

    def load_agents(self) -> None:
        """Load and parse all agent configuration files."""
        agent_files = sorted(self.agents_dir.glob("*.md"))

        for agent_file in agent_files:
            agent_data = self._parse_agent_file(agent_file)
            if agent_data:
                self.agents.append(agent_data)

    def _parse_agent_file(self, file_path: Path) -> Dict:
        """Parse a single agent configuration file.

        Args:
            file_path: Path to agent markdown file

        Returns:
            Dictionary containing agent metadata and content
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract YAML frontmatter
            frontmatter = {}
            yaml_match = re.search(r"^---\n(.*?)\n---", content, re.DOTALL)
            if yaml_match:
                yaml_content = yaml_match.group(1)
                for line in yaml_content.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        frontmatter[key.strip()] = value.strip()

            # Extract level from content
            level = None
            level_match = re.search(r"Level (\d)", content)
            if level_match:
                level = int(level_match.group(1))

            # Extract delegation links
            delegations = re.findall(r"\[([^\]]+)\]\(\./([^)]+\.md)\)", content)

            # Extract tools from frontmatter
            tools = []
            if "tools" in frontmatter:
                tools = [t.strip() for t in frontmatter["tools"].split(",")]

            # Extract skills mentioned in content
            skills = re.findall(r"`([a-z-]+)`\s+skill", content.lower())

            return {
                "name": file_path.stem,
                "file": file_path.name,
                "frontmatter": frontmatter,
                "level": level,
                "tools": tools,
                "skills": list(set(skills)),
                "delegations": delegations,
                "content_length": len(content),
            }
        except Exception as e:
            print(f"Error parsing {file_path}: {e}", file=sys.stderr)
            return None

    def analyze(self) -> None:
        """Perform statistical analysis on loaded agents."""
        self.stats = {
            "total_agents": len(self.agents),
            "by_level": defaultdict(list),
            "by_tool": defaultdict(list),
            "by_skill": defaultdict(list),
            "delegation_graph": defaultdict(list),
            "tool_frequency": defaultdict(int),
            "skill_frequency": defaultdict(int),
            "agents_without_level": [],
        }

        for agent in self.agents:
            # Count by level
            if agent["level"] is not None:
                self.stats["by_level"][agent["level"]].append(agent["name"])
            else:
                self.stats["agents_without_level"].append(agent["name"])

            # Count by tool
            for tool in agent["tools"]:
                self.stats["by_tool"][tool].append(agent["name"])
                self.stats["tool_frequency"][tool] += 1

            # Count by skill
            for skill in agent["skills"]:
                self.stats["by_skill"][skill].append(agent["name"])
                self.stats["skill_frequency"][skill] += 1

            # Build delegation graph
            for link_text, link_file in agent["delegations"]:
                target = link_file.replace(".md", "")
                self.stats["delegation_graph"][agent["name"]].append({"target": target, "description": link_text})

    def format_text(self) -> str:
        """Format statistics as plain text report.

        Returns:
            Plain text formatted report
        """
        lines = []
        lines.append("=" * 60)
        lines.append("Agent System Statistics Report")
        lines.append("=" * 60)
        lines.append("")

        # Overview
        lines.append("OVERVIEW")
        lines.append("-" * 60)
        lines.append(f"Total Agents: {self.stats['total_agents']}")
        lines.append("")

        # By Level
        lines.append("AGENTS BY LEVEL")
        lines.append("-" * 60)
        for level in sorted(self.stats["by_level"].keys()):
            agents = self.stats["by_level"][level]
            lines.append(f"Level {level}: {len(agents)} agents")
            if self.verbose:
                for agent in sorted(agents):
                    lines.append(f"  - {agent}")
        if self.stats["agents_without_level"]:
            lines.append(f"No level: {len(self.stats['agents_without_level'])} agents")
            if self.verbose:
                for agent in sorted(self.stats["agents_without_level"]):
                    lines.append(f"  - {agent}")
        lines.append("")

        # Tool Usage
        lines.append("TOOL USAGE")
        lines.append("-" * 60)
        sorted_tools = sorted(self.stats["tool_frequency"].items(), key=lambda x: x[1], reverse=True)
        for tool, count in sorted_tools:
            lines.append(f"{tool}: {count} agents")
            if self.verbose:
                for agent in sorted(self.stats["by_tool"][tool]):
                    lines.append(f"  - {agent}")
        lines.append("")

        # Skill References
        lines.append("SKILL REFERENCES")
        lines.append("-" * 60)
        if self.stats["skill_frequency"]:
            sorted_skills = sorted(self.stats["skill_frequency"].items(), key=lambda x: x[1], reverse=True)
            for skill, count in sorted_skills:
                lines.append(f"{skill}: {count} references")
                if self.verbose:
                    for agent in sorted(self.stats["by_skill"][skill]):
                        lines.append(f"  - {agent}")
        else:
            lines.append("No skill references found")
        lines.append("")

        # Delegation Patterns
        lines.append("DELEGATION PATTERNS")
        lines.append("-" * 60)
        agents_with_delegations = {k: v for k, v in self.stats["delegation_graph"].items() if v}
        lines.append(f"Agents that delegate: {len(agents_with_delegations)}")
        if self.verbose and agents_with_delegations:
            for agent, delegations in sorted(agents_with_delegations.items()):
                lines.append(f"\n{agent}:")
                for delegation in delegations:
                    lines.append(f"  → {delegation['target']}")
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 60)
        lines.append(f"Total unique tools: {len(self.stats['tool_frequency'])}")
        lines.append(f"Total unique skills: {len(self.stats['skill_frequency'])}")
        lines.append(f"Total delegation links: {sum(len(v) for v in self.stats['delegation_graph'].values())}")
        lines.append("")

        return "\n".join(lines)

    def format_markdown(self) -> str:
        """Format statistics as markdown report.

        Returns:
            Markdown formatted report
        """
        lines = []
        lines.append("# Agent System Statistics Report")
        lines.append("")

        # Overview
        lines.append("## Overview")
        lines.append("")
        lines.append(f"**Total Agents**: {self.stats['total_agents']}")
        lines.append("")

        # By Level
        lines.append("## Agents by Level")
        lines.append("")
        lines.append("| Level | Count | Agents |")
        lines.append("|-------|-------|--------|")
        for level in sorted(self.stats["by_level"].keys()):
            agents = self.stats["by_level"][level]
            agent_list = ", ".join(sorted(agents)) if self.verbose else ""
            lines.append(f"| {level} | {len(agents)} | {agent_list} |")
        if self.stats["agents_without_level"]:
            agent_list = ", ".join(sorted(self.stats["agents_without_level"])) if self.verbose else ""
            lines.append(f"| N/A | {len(self.stats['agents_without_level'])} | {agent_list} |")
        lines.append("")

        # Tool Usage
        lines.append("## Tool Usage")
        lines.append("")
        sorted_tools = sorted(self.stats["tool_frequency"].items(), key=lambda x: x[1], reverse=True)
        lines.append("| Tool | Usage Count |")
        lines.append("|------|-------------|")
        for tool, count in sorted_tools:
            lines.append(f"| {tool} | {count} |")
        lines.append("")

        # Skill References
        lines.append("## Skill References")
        lines.append("")
        if self.stats["skill_frequency"]:
            sorted_skills = sorted(self.stats["skill_frequency"].items(), key=lambda x: x[1], reverse=True)
            lines.append("| Skill | Reference Count |")
            lines.append("|-------|----------------|")
            for skill, count in sorted_skills:
                lines.append(f"| {skill} | {count} |")
        else:
            lines.append("*No skill references found*")
        lines.append("")

        # Delegation Patterns
        lines.append("## Delegation Patterns")
        lines.append("")
        agents_with_delegations = {k: v for k, v in self.stats["delegation_graph"].items() if v}
        lines.append(f"**Agents that delegate**: {len(agents_with_delegations)}")
        lines.append("")

        if self.verbose and agents_with_delegations:
            for agent, delegations in sorted(agents_with_delegations.items()):
                lines.append(f"### {agent}")
                lines.append("")
                for delegation in delegations:
                    lines.append(f"- {delegation['target']}")
                lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total unique tools**: {len(self.stats['tool_frequency'])}")
        lines.append(f"- **Total unique skills**: {len(self.stats['skill_frequency'])}")
        lines.append(f"- **Total delegation links**: {sum(len(v) for v in self.stats['delegation_graph'].values())}")
        lines.append("")

        return "\n".join(lines)

    def format_json(self) -> str:
        """Format statistics as JSON.

        Returns:
            JSON formatted report
        """
        # Convert defaultdict to regular dict for JSON serialization
        output = {
            "total_agents": self.stats["total_agents"],
            "by_level": {str(k): v for k, v in self.stats["by_level"].items()},
            "tool_usage": dict(self.stats["tool_frequency"]),
            "skill_references": dict(self.stats["skill_frequency"]),
            "delegation_count": len([k for k, v in self.stats["delegation_graph"].items() if v]),
            "total_delegations": sum(len(v) for v in self.stats["delegation_graph"].values()),
        }

        if self.verbose:
            output["by_tool"] = dict(self.stats["by_tool"])
            output["by_skill"] = dict(self.stats["by_skill"])
            output["delegation_graph"] = dict(self.stats["delegation_graph"])
            output["agents_without_level"] = self.stats["agents_without_level"]

        return json.dumps(output, indent=2)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate usage statistics for the agent system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate text report to stdout
  python3 scripts/agents/agent_stats.py

  # Generate markdown report to file
  python3 scripts/agents/agent_stats.py --format markdown --output stats.md

  # Generate JSON with verbose details
  python3 scripts/agents/agent_stats.py --format json --verbose
        """,
    )

    parser.add_argument(
        "--format", choices=["text", "json", "markdown"], default="text", help="Output format (default: text)"
    )
    parser.add_argument("--output", type=str, help="Write output to file (default: stdout)")
    parser.add_argument("--verbose", action="store_true", help="Include detailed breakdowns")

    args = parser.parse_args()

    # Find repository root and agents directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    agents_dir = repo_root / ".claude" / "agents"

    if not agents_dir.exists():
        print(f"Error: Agents directory not found: {agents_dir}", file=sys.stderr)
        return 1

    # Run analysis
    analyzer = AgentAnalyzer(agents_dir, verbose=args.verbose)
    analyzer.load_agents()

    if not analyzer.agents:
        print("Error: No agent files found", file=sys.stderr)
        return 1

    analyzer.analyze()

    # Format output
    if args.format == "text":
        output = analyzer.format_text()
    elif args.format == "markdown":
        output = analyzer.format_markdown()
    elif args.format == "json":
        output = analyzer.format_json()
    else:
        print(f"Error: Unknown format: {args.format}", file=sys.stderr)
        return 1

    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Report written to: {output_path}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
