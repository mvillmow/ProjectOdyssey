#!/usr/bin/env python3
"""Shared utilities for agent configuration scripts.

This module provides common functions for working with agent markdown files
that contain YAML frontmatter.
"""

import re
from typing import Dict, Optional, Tuple

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
