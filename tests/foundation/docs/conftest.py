"""
Shared pytest fixtures for documentation test suite.

This module provides common fixtures used across all documentation tests
to eliminate code duplication and ensure consistency.

Fixtures:
- repo_root: Real repository root directory
- docs_root: Real docs/ directory
- tier_directories: All tier directory paths
- all_doc_files: All documentation files paths
"""

import pytest
from pathlib import Path
from typing import List, Dict


# Constants for test configuration
MIN_DOC_LENGTH = 100
MAX_LINE_LENGTH = 120
TIER_NAMES = ["getting-started", "core", "advanced", "dev"]
ROOT_DOCS = ["README.md"]


@pytest.fixture
def repo_root() -> Path:
    """
    Provide the real repository root directory for testing.

    Returns:
        Path to the actual repository root directory
    """
    # Navigate up from tests/foundation/docs/ to repository root
    current_file = Path(__file__)
    return current_file.parent.parent.parent.parent


@pytest.fixture
def docs_root(repo_root: Path) -> Path:
    """
    Provide the expected docs directory path (may not exist yet).

    Args:
        repo_root: Real repository root directory

    Returns:
        Path to docs directory within repository root
    """
    return repo_root / "docs"


@pytest.fixture
def tier_directories(docs_root: Path) -> Dict[str, Path]:
    """
    Provide paths to all tier directories (may not exist yet).

    Args:
        docs_root: Path to docs directory

    Returns:
        Dictionary mapping tier names to their paths
    """
    return {tier_name: docs_root / tier_name for tier_name in TIER_NAMES}


@pytest.fixture
def getting_started_dir(docs_root: Path) -> Path:
    """
    Provide the getting-started documentation directory path (may not exist yet).

    Args:
        docs_root: Path to docs directory

    Returns:
        Path to docs/getting-started directory
    """
    return docs_root / "getting-started"


@pytest.fixture
def getting_started_docs(tier_directories: Dict[str, Path]) -> List[Path]:
    """
    Provide list of Getting Started (Tier 1) documentation files.

    Args:
        tier_directories: Dictionary of tier directory paths

    Returns:
        List of paths to Tier 1 documentation files
    """
    tier1_dir = tier_directories["getting-started"]
    doc_files = [
        "README.md",
        "installation.md",
        "quick-start.md",
        "first-steps.md",
        "core-concepts.md",
        "tutorials.md",
    ]
    return [tier1_dir / doc for doc in doc_files]


@pytest.fixture
def core_docs(tier_directories: Dict[str, Path]) -> List[Path]:
    """
    Provide list of Core Documentation (Tier 2) files.

    Args:
        tier_directories: Dictionary of tier directory paths

    Returns:
        List of paths to Tier 2 documentation files
    """
    tier2_dir = tier_directories["core"]
    doc_files = [
        "README.md",
        "architecture.md",
        "api-reference.md",
        "data-structures.md",
        "algorithms.md",
        "configuration.md",
        "modules.md",
        "utilities.md",
    ]
    return [tier2_dir / doc for doc in doc_files]


@pytest.fixture
def advanced_docs(tier_directories: Dict[str, Path]) -> List[Path]:
    """
    Provide list of Advanced Topics (Tier 3) documentation files.

    Args:
        tier_directories: Dictionary of tier directory paths

    Returns:
        List of paths to Tier 3 documentation files
    """
    tier3_dir = tier_directories["advanced"]
    doc_files = [
        "README.md",
        "optimization.md",
        "custom-ops.md",
        "distributed.md",
        "performance.md",
        "advanced-patterns.md",
    ]
    return [tier3_dir / doc for doc in doc_files]


@pytest.fixture
def dev_docs(tier_directories: Dict[str, Path]) -> List[Path]:
    """
    Provide list of Development Guides (Tier 4) documentation files.

    Args:
        tier_directories: Dictionary of tier directory paths

    Returns:
        List of paths to Tier 4 documentation files
    """
    tier4_dir = tier_directories["dev"]
    doc_files = [
        "README.md",
        "contributing.md",
        "testing.md",
        "ci-cd.md",
    ]
    return [tier4_dir / doc for doc in doc_files]


@pytest.fixture
def all_doc_files(
    getting_started_docs: List[Path],
    core_docs: List[Path],
    advanced_docs: List[Path],
    dev_docs: List[Path],
    repo_root: Path,
) -> List[Path]:
    """
    Provide list of all documentation files across all tiers.

    Args:
        getting_started_docs: Tier 1 documentation files
        core_docs: Tier 2 documentation files
        advanced_docs: Tier 3 documentation files
        dev_docs: Tier 4 documentation files
        repo_root: Repository root path

    Returns:
        List of all documentation file paths
    """
    all_docs = getting_started_docs + core_docs + advanced_docs + dev_docs + [repo_root / doc for doc in ROOT_DOCS]
    return all_docs
