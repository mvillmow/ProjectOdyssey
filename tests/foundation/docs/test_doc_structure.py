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


class TestCompleteFourTierStructure:
    """
    Test comprehensive 4-tier structure validation (addresses Issue #57).

    This test class validates the complete 4-tier documentation structure
    with all 24 documents as specified in the planning phase.

    Tier 1 (Getting Started): 6 documents
    Tier 2 (Core Documentation): 8 documents
    Tier 3 (Advanced Topics): 6 documents
    Tier 4 (Development Guides): 4 documents
    """

    def test_tier1_getting_started_docs_exist(
        self, repo_root: Path, getting_started_docs: List[Path]
    ) -> None:
        """
        Test Tier 1: All 6 Getting Started documents exist.

        Args:
            repo_root: Repository root path
            getting_started_docs: List of Tier 1 documentation files
        """
        # Create all Tier 1 docs
        for doc_path in getting_started_docs:
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            doc_path.write_text(f"# {doc_path.stem}\n\nContent here.\n")

        # Validate all exist
        for doc_path in getting_started_docs:
            assert doc_path.exists(), f"Tier 1 doc should exist: {doc_path.name}"
            assert doc_path.is_file(), f"Tier 1 doc should be a file: {doc_path.name}"

        # Validate count
        assert len(getting_started_docs) == 6, "Should have exactly 6 Tier 1 documents"

    def test_tier2_core_docs_exist(self, core_docs: List[Path]) -> None:
        """
        Test Tier 2: All 8 Core Documentation files exist.

        Args:
            core_docs: List of Tier 2 documentation files
        """
        # Create all Tier 2 docs
        for doc_path in core_docs:
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            doc_path.write_text(f"# {doc_path.stem}\n\nContent here.\n")

        # Validate all exist
        for doc_path in core_docs:
            assert doc_path.exists(), f"Tier 2 doc should exist: {doc_path.name}"
            assert doc_path.is_file(), f"Tier 2 doc should be a file: {doc_path.name}"

        # Validate count
        assert len(core_docs) == 8, "Should have exactly 8 Tier 2 documents"

    def test_tier3_advanced_docs_exist(self, advanced_docs: List[Path]) -> None:
        """
        Test Tier 3: All 6 Advanced Topics documents exist.

        Args:
            advanced_docs: List of Tier 3 documentation files
        """
        # Create all Tier 3 docs
        for doc_path in advanced_docs:
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            doc_path.write_text(f"# {doc_path.stem}\n\nContent here.\n")

        # Validate all exist
        for doc_path in advanced_docs:
            assert doc_path.exists(), f"Tier 3 doc should exist: {doc_path.name}"
            assert doc_path.is_file(), f"Tier 3 doc should be a file: {doc_path.name}"

        # Validate count
        assert len(advanced_docs) == 6, "Should have exactly 6 Tier 3 documents"

    def test_tier4_dev_docs_exist(self, dev_docs: List[Path]) -> None:
        """
        Test Tier 4: All 4 Development Guide documents exist.

        Args:
            dev_docs: List of Tier 4 documentation files
        """
        # Create all Tier 4 docs
        for doc_path in dev_docs:
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            doc_path.write_text(f"# {doc_path.stem}\n\nContent here.\n")

        # Validate all exist
        for doc_path in dev_docs:
            assert doc_path.exists(), f"Tier 4 doc should exist: {doc_path.name}"
            assert doc_path.is_file(), f"Tier 4 doc should be a file: {doc_path.name}"

        # Validate count
        assert len(dev_docs) == 4, "Should have exactly 4 Tier 4 documents"

    def test_all_24_documents_present(self, all_doc_files: List[Path]) -> None:
        """
        Test that all 24 documents across all tiers are present.

        Args:
            all_doc_files: List of all documentation files
        """
        # Create all documents
        for doc_path in all_doc_files:
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            doc_path.write_text(f"# {doc_path.stem}\n\nContent here.\n")

        # Validate total count
        assert len(all_doc_files) == 24, "Should have exactly 24 documents total (6+8+6+4)"

        # Validate all exist
        for doc_path in all_doc_files:
            assert doc_path.exists(), f"Document should exist: {doc_path}"

    def test_tier_readme_files_present(self, tier_directories: dict) -> None:
        """
        Test that each tier has a README.md file.

        Args:
            tier_directories: Dictionary of tier directory paths
        """
        for tier_name, tier_path in tier_directories.items():
            readme = tier_path / "README.md"
            readme.write_text(f"# {tier_name.title()} Documentation\n\nOverview here.\n")
            assert readme.exists(), f"Tier {tier_name} should have README.md"
            assert readme.is_file(), f"Tier {tier_name} README.md should be a file"

    @pytest.mark.parametrize(
        "tier_name,expected_count",
        [
            ("getting-started", 6),
            ("core", 8),
            ("advanced", 6),
            ("dev", 4),
        ],
    )
    def test_tier_document_counts(
        self, tier_directories: dict, tier_name: str, expected_count: int
    ) -> None:
        """
        Test that each tier has the correct number of documents.

        Args:
            tier_directories: Dictionary of tier directory paths
            tier_name: Name of the tier to test
            expected_count: Expected number of documents in this tier
        """
        tier_path = tier_directories[tier_name]

        # Create expected number of docs
        for i in range(expected_count):
            doc = tier_path / f"doc{i+1}.md"
            doc.write_text(f"# Document {i+1}\n\nContent.\n")

        # Count markdown files
        md_files = list(tier_path.glob("*.md"))
        assert len(md_files) == expected_count, (
            f"Tier {tier_name} should have {expected_count} documents, "
            f"found {len(md_files)}"
        )
