#!/usr/bin/env python3
"""Shared utilities for agent configuration scripts.

This module provides common functions for working with agent markdown files
that contain YAML frontmatter.

Functions:
    extract_frontmatter_raw: Extract YAML frontmatter text only
    extract_frontmatter_with_lines: Extract with line number tracking
    extract_frontmatter_parsed: Extract and parse to dictionary
    find_agent_files: Discover agent markdown files
    load_agent: Load a single agent configuration
    load_all_agents: Load all agent configurations from a directory
    validate_frontmatter_structure: Validate frontmatter structure

Classes:
    AgentInfo: Container for agent metadata
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Regex pattern for YAML frontmatter (shared across all variants)
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?\n)---\s*\n", re.DOTALL)


def extract_frontmatter_raw(content: str) -> Optional[str]:
    """Extract frontmatter text only.

    Args:
        content: Markdown file content.

    Returns:
        The raw YAML frontmatter string, or None if not found.
    """
    match = FRONTMATTER_PATTERN.match(content)
    if match:
        return match.group(1)
    return None


def extract_frontmatter_with_lines(content: str) -> Optional[Tuple[str, int, int]]:
    """Extract frontmatter with line number tracking.

    Args:
        content: Markdown file content.

    Returns:
        Tuple of (frontmatter_text, start_line, end_line), or None if not found.
        Lines are 1-indexed, counting from the beginning of the file.
    """
    match = FRONTMATTER_PATTERN.match(content)
    if match:
        frontmatter = match.group(1)
        # start_line is line 1 (the --- marker), end_line is where the closing --- is
        start_line = 1
        end_line = content[: match.end()].count("\n")
        return (frontmatter, start_line, end_line)
    return None


def extract_frontmatter_parsed(content: str) -> Optional[Tuple[str, Dict]]:
    """Extract and parse frontmatter to Dict.

    Args:
        content: Markdown file content.

    Returns:
        Tuple of (frontmatter_text, parsed_dict), or None if not found or invalid.
    """
    match = FRONTMATTER_PATTERN.match(content)
    if match:
        frontmatter = match.group(1)
        try:
            parsed = yaml.safe_load(frontmatter)
            if isinstance(parsed, dict):
                return (frontmatter, parsed)
        except yaml.YAMLError:
            pass
    return None


class AgentInfo:
    """Information about an agent configuration.

    Attributes:
        file_path: Path to the agent markdown file
        name: Agent name from frontmatter
        description: Agent description from frontmatter
        tools: Comma-separated tools string from frontmatter
        model: Model assignment (sonnet, opus, haiku, etc.)
        level: Agent level (0-5) in the hierarchy
    """

    def __init__(self, file_path: Path, frontmatter: Dict):
        """Initialize AgentInfo from file path and parsed frontmatter.

        Args:
            file_path: Path to the markdown file
            frontmatter: Parsed YAML frontmatter dictionary
        """
        self.file_path = file_path
        self.name = frontmatter.get("name", "unknown")
        self.description = frontmatter.get("description", "No description")
        self.tools = frontmatter.get("tools", "")
        self.model = frontmatter.get("model", "unknown")
        self.level = self._infer_level(frontmatter)

    def _infer_level(self, frontmatter: Dict) -> int:
        """Infer agent level from frontmatter or name.

        Level hierarchy:
        - 0: Meta-orchestrator (Chief Architect)
        - 1: Section Orchestrators
        - 2: Design Agents
        - 3: Component Specialists
        - 4: Senior Engineers
        - 5: Junior Engineers

        Args:
            frontmatter: Parsed frontmatter dictionary

        Returns:
            Agent level (0-5)
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
        """Get list of tool names.

        Returns:
            List of individual tool names
        """
        if not self.tools:
            return []
        return [t.strip() for t in self.tools.split(",")]

    def __repr__(self):
        """Return string representation."""
        return f"AgentInfo(level={self.level}, name={self.name})"


def find_agent_files(agents_dir: Path) -> List[Path]:
    """Find all agent markdown files in a directory.

    Args:
        agents_dir: Path to agents directory

    Returns:
        Sorted list of agent markdown file paths
    """
    return sorted(agents_dir.glob("*.md"))


def load_agent(file_path: Path) -> Optional[AgentInfo]:
    """Load agent configuration from a markdown file.

    Args:
        file_path: Path to the agent markdown file

    Returns:
        AgentInfo object or None if loading failed
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return None

    frontmatter_text = extract_frontmatter_raw(content)
    if frontmatter_text is None:
        return None

    try:
        frontmatter = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError:
        return None

    if not isinstance(frontmatter, dict):
        return None

    return AgentInfo(file_path, frontmatter)


def load_all_agents(agents_dir: Path) -> List[AgentInfo]:
    """Load all agent configurations from a directory.

    Args:
        agents_dir: Path to agents directory

    Returns:
        List of AgentInfo objects
    """
    agent_files = find_agent_files(agents_dir)
    agents = []

    for file_path in agent_files:
        agent = load_agent(file_path)
        if agent:
            agents.append(agent)

    return agents


def validate_frontmatter_structure(
    frontmatter: Dict,
    required_fields: Optional[Dict[str, type]] = None,
    optional_fields: Optional[Dict[str, type]] = None,
) -> List[str]:
    """Validate frontmatter structure.

    Args:
        frontmatter: Parsed YAML frontmatter dictionary
        required_fields: Dictionary mapping field names to expected types
        optional_fields: Dictionary mapping optional field names to expected types

    Returns:
        List of error messages (empty if valid)
    """
    if required_fields is None:
        required_fields = {
            "name": str,
            "description": str,
            "tools": str,
            "model": str,
        }

    if optional_fields is None:
        optional_fields = {
            "level": int,
            "section": str,
            "workflow_phase": str,
        }

    errors = []

    # Check required fields
    for field, expected_type in required_fields.items():
        if field not in frontmatter:
            errors.append(f"Missing required field: '{field}'")
        else:
            value = frontmatter[field]
            if not isinstance(value, expected_type):
                errors.append(f"Field '{field}' should be {expected_type.__name__}, got {type(value).__name__}")

    # Check optional fields if present
    for field, expected_type in optional_fields.items():
        if field in frontmatter:
            value = frontmatter[field]
            if not isinstance(value, expected_type):
                errors.append(f"Field '{field}' should be {expected_type.__name__}, got {type(value).__name__}")

    return errors
