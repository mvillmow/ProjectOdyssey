"""
Shared fixtures for tools/ directory infrastructure tests.

This module provides reusable test fixtures for validating the tools directory
structure and documentation.
"""

import pytest
from pathlib import Path
from typing import Dict


@pytest.fixture
def repo_root() -> Path:
    """
    Provide the repository root directory path.

    Returns:
        Path to repository root
    """
    # Navigate up from tests/tooling/tools/ to repository root
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent.parent


@pytest.fixture
def tools_root(repo_root: Path) -> Path:
    """
    Provide path to tools/ directory.

    Args:
        repo_root: Repository root directory

    Returns:
        Path to tools/ directory
    """
    return repo_root / "tools"


@pytest.fixture
def category_names() -> list:
    """
    Provide list of expected tool category names.

    Returns:
        List of category directory names
    """
    return ["paper-scaffold", "test-utils", "benchmarking", "codegen"]


@pytest.fixture
def category_paths(tools_root: Path, category_names: list) -> Dict[str, Path]:
    """
    Provide dictionary mapping category names to their directory paths.

    Args:
        tools_root: Path to tools/ directory
        category_names: List of category names

    Returns:
        Dictionary of {category_name: category_path}
    """
    return {name: tools_root / name for name in category_names}


@pytest.fixture
def main_readme(tools_root: Path) -> Path:
    """
    Provide path to main tools/README.md file.

    Args:
        tools_root: Path to tools/ directory

    Returns:
        Path to tools/README.md
    """
    return tools_root / "README.md"


@pytest.fixture
def category_readmes(category_paths: Dict[str, Path]) -> Dict[str, Path]:
    """
    Provide dictionary mapping category names to their README.md paths.

    Args:
        category_paths: Dictionary of category paths

    Returns:
        Dictionary of {category_name: readme_path}
    """
    return {name: path / "README.md" for name, path in category_paths.items()}
