#!/usr/bin/env python3
"""
Shared utilities and constants for ML Odyssey scripts

This module provides common functionality used across multiple scripts to avoid duplication.
"""

from pathlib import Path
from typing import Optional


# Label colors for GitHub issues (5-phase development workflow)
# Used by: create_issues.py, create_single_component_issues.py
LABEL_COLORS = {
    'planning': 'd4c5f9',       # Light purple
    'documentation': '0075ca',  # Blue
    'testing': 'fbca04',        # Yellow
    'tdd': 'fbca04',           # Yellow
    'implementation': '1d76db', # Dark blue
    'packaging': 'c2e0c6',      # Light green
    'integration': 'c2e0c6',    # Light green
    'cleanup': 'd93f0b'         # Red
}


def get_repo_root() -> Path:
    """
    Get the repository root directory.

    Searches upward from the current file location until finding a directory
    containing a .git folder.

    Returns:
        Path to repository root

    Raises:
        RuntimeError: If repository root cannot be found
    """
    current = Path(__file__).resolve().parent

    # Search upward for .git directory
    while current != current.parent:
        if (current / '.git').exists():
            return current
        current = current.parent

    raise RuntimeError("Could not find repository root (no .git directory found)")


def get_agents_dir() -> Path:
    """
    Get the .claude/agents directory path.

    Returns:
        Path to .claude/agents directory

    Raises:
        RuntimeError: If agents directory doesn't exist
    """
    repo_root = get_repo_root()
    agents_dir = repo_root / '.claude' / 'agents'

    if not agents_dir.exists():
        raise RuntimeError(f"Agents directory not found: {agents_dir}")

    return agents_dir


def get_plan_dir() -> Path:
    """
    Get the notes/plan directory path.

    Returns:
        Path to notes/plan directory

    Raises:
        RuntimeError: If plan directory doesn't exist
    """
    repo_root = get_repo_root()
    plan_dir = repo_root / 'notes' / 'plan'

    if not plan_dir.exists():
        raise RuntimeError(f"Plan directory not found: {plan_dir}")

    return plan_dir
