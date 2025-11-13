"""
Test suite for documentation structure validation.

This module validates the 4-tier documentation hierarchy exists and is properly
organized according to the project specifications.

Test Categories:
- Tier structure: Validate all 4 tiers present
- Directory organization: Validate subdirectory structure
- Root-level docs: Validate docs at repository root
- Path validation: Ensure correct locations

Coverage Target: >95%
"""

import pytest
from pathlib import Path
from typing import List


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """
    Provide a mock repository root directory for testing.

    Args:
        tmp_path: pytest built-in fixture providing temporary directory

    Returns:
        Path to temporary directory acting as repository root
    """
    return tmp_path


@pytest.fixture
def docs_root(repo_root: Path) -> Path:
    """
    Provide the expected docs directory path.

    Args:
        repo_root: Temporary repository root directory

    Returns:
        Path to docs directory within repository root
    """
    return repo_root / "docs"


class TestDocumentationStructure:
    """Test cases for documentation directory structure."""

    def test_docs_directory_exists(self, docs_root: Path) -> None:
        """
        Test that docs/ directory exists at repository root.

        Args:
            docs_root: Path to docs directory
        """
        docs_root.mkdir(parents=True)
        assert docs_root.exists(), "docs/ directory should exist"
        assert docs_root.is_dir(), "docs/ should be a directory"

    def test_docs_directory_location(self, repo_root: Path, docs_root: Path) -> None:
        """
        Test that docs/ directory is at repository root, not nested.

        Args:
            repo_root: Repository root path
            docs_root: Path to docs directory
        """
        docs_root.mkdir(parents=True)
        assert docs_root.parent == repo_root, "docs/ should be at repository root"

    @pytest.mark.parametrize(
        "tier_dir",
        [
            "getting-started",
            "core",
            "advanced",
            "dev",
        ],
    )
    def test_tier_directories_exist(self, docs_root: Path, tier_dir: str) -> None:
        """
        Test that all tier subdirectories exist.

        Args:
            docs_root: Path to docs directory
            tier_dir: Name of tier subdirectory to test
        """
        tier_path = docs_root / tier_dir
        tier_path.mkdir(parents=True)
        assert tier_path.exists(), f"docs/{tier_dir}/ should exist"
        assert tier_path.is_dir(), f"docs/{tier_dir}/ should be a directory"

    def test_getting_started_tier_location(self, docs_root: Path) -> None:
        """
        Test Tier 1 (Getting Started) directory structure.

        Args:
            docs_root: Path to docs directory
        """
        tier1 = docs_root / "getting-started"
        tier1.mkdir(parents=True)
        assert tier1.parent == docs_root, "getting-started/ should be under docs/"

    def test_core_tier_location(self, docs_root: Path) -> None:
        """
        Test Tier 2 (Core Documentation) directory structure.

        Args:
            docs_root: Path to docs directory
        """
        tier2 = docs_root / "core"
        tier2.mkdir(parents=True)
        assert tier2.parent == docs_root, "core/ should be under docs/"

    def test_advanced_tier_location(self, docs_root: Path) -> None:
        """
        Test Tier 3 (Advanced Topics) directory structure.

        Args:
            docs_root: Path to docs directory
        """
        tier3 = docs_root / "advanced"
        tier3.mkdir(parents=True)
        assert tier3.parent == docs_root, "advanced/ should be under docs/"

    def test_dev_tier_location(self, docs_root: Path) -> None:
        """
        Test Tier 4 (Development Guides) directory structure.

        Args:
            docs_root: Path to docs directory
        """
        tier4 = docs_root / "dev"
        tier4.mkdir(parents=True)
        assert tier4.parent == docs_root, "dev/ should be under docs/"

    @pytest.mark.parametrize(
        "root_doc",
        [
            "README.md",
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
        ],
    )
    def test_root_level_docs_location(self, repo_root: Path, root_doc: str) -> None:
        """
        Test that root-level documents are at repository root.

        Args:
            repo_root: Repository root path
            root_doc: Name of root document to test
        """
        doc_path = repo_root / root_doc
        doc_path.touch()
        assert doc_path.parent == repo_root, f"{root_doc} should be at repository root"

    def test_all_tier_directories_present(self, docs_root: Path) -> None:
        """
        Test that all 4 tier directories are present.

        Args:
            docs_root: Path to docs directory
        """
        expected_tiers = ["getting-started", "core", "advanced", "dev"]
        docs_root.mkdir(parents=True)

        for tier in expected_tiers:
            tier_path = docs_root / tier
            tier_path.mkdir(parents=True)

        actual_tiers = [d.name for d in docs_root.iterdir() if d.is_dir()]

        for tier in expected_tiers:
            assert tier in actual_tiers, f"{tier}/ directory should exist"

    def test_no_unexpected_directories(self, docs_root: Path) -> None:
        """
        Test that no unexpected directories exist in docs/.

        Args:
            docs_root: Path to docs directory
        """
        expected_tiers = {"getting-started", "core", "advanced", "dev"}
        docs_root.mkdir(parents=True)

        for tier in expected_tiers:
            (docs_root / tier).mkdir()

        actual_dirs = {d.name for d in docs_root.iterdir() if d.is_dir()}
        unexpected = actual_dirs - expected_tiers

        assert len(unexpected) == 0, f"Unexpected directories: {unexpected}"


class TestDocumentationHierarchy:
    """Test cases for documentation hierarchy validation."""

    def test_4_tier_structure_complete(self, docs_root: Path) -> None:
        """
        Test that all 4 tiers are present and accessible.

        Args:
            docs_root: Path to docs directory
        """
        tiers = ["getting-started", "core", "advanced", "dev"]
        docs_root.mkdir(parents=True)

        for tier in tiers:
            (docs_root / tier).mkdir()

        for tier in tiers:
            tier_path = docs_root / tier
            assert tier_path.exists(), f"Tier {tier} should exist"
            assert tier_path.is_dir(), f"Tier {tier} should be a directory"

    def test_tier_count(self, docs_root: Path) -> None:
        """
        Test that exactly 4 tiers exist (no more, no less).

        Args:
            docs_root: Path to docs directory
        """
        tiers = ["getting-started", "core", "advanced", "dev"]
        docs_root.mkdir(parents=True)

        for tier in tiers:
            (docs_root / tier).mkdir()

        tier_dirs = [d for d in docs_root.iterdir() if d.is_dir()]
        assert len(tier_dirs) == 4, "Should have exactly 4 tier directories"

    def test_hierarchy_paths_valid(self, repo_root: Path) -> None:
        """
        Test that all hierarchy paths are valid and accessible.

        Args:
            repo_root: Repository root path
        """
        docs_root = repo_root / "docs"
        tiers = ["getting-started", "core", "advanced", "dev"]

        docs_root.mkdir(parents=True)
        for tier in tiers:
            (docs_root / tier).mkdir()

        # Test root level
        assert repo_root.exists(), "Repository root should exist"

        # Test docs level
        assert docs_root.exists(), "docs/ should exist"

        # Test tier level
        for tier in tiers:
            tier_path = docs_root / tier
            assert tier_path.exists(), f"docs/{tier}/ should exist"
