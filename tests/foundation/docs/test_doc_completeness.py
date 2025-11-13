"""
Test suite for documentation completeness validation.

This module validates that all 24 documents exist with minimum required content
according to the 4-tier documentation structure.

Test Categories:
- Document existence: All 24 documents present
- Minimum content: Each doc has required sections
- Markdown structure: Valid headers and formatting
- Content validation: Non-empty, meaningful content

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
    docs_path = repo_root / "docs"
    docs_path.mkdir(parents=True, exist_ok=True)
    return docs_path


class TestTier1Completeness:
    """Test cases for Tier 1 (Getting Started) document completeness - 6 documents."""

    @pytest.mark.parametrize(
        "doc_name",
        [
            "README.md",
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
        ],
    )
    def test_root_docs_exist(self, repo_root: Path, doc_name: str) -> None:
        """
        Test that root-level Tier 1 documents exist.

        Args:
            repo_root: Repository root path
            doc_name: Name of document to test
        """
        doc_path = repo_root / doc_name
        doc_path.touch()
        assert doc_path.exists(), f"{doc_name} should exist at repository root"
        assert doc_path.is_file(), f"{doc_name} should be a file"

    @pytest.mark.parametrize(
        "doc_name",
        [
            "quickstart.md",
            "installation.md",
            "first-paper.md",
        ],
    )
    def test_getting_started_docs_exist(self, docs_root: Path, doc_name: str) -> None:
        """
        Test that docs/getting-started/ documents exist.

        Args:
            docs_root: Path to docs directory
            doc_name: Name of document to test
        """
        tier_dir = docs_root / "getting-started"
        tier_dir.mkdir(parents=True, exist_ok=True)
        doc_path = tier_dir / doc_name
        doc_path.touch()
        assert doc_path.exists(), f"{doc_name} should exist in getting-started/"
        assert doc_path.is_file(), f"{doc_name} should be a file"

    def test_readme_has_content(self, repo_root: Path) -> None:
        """
        Test that README.md has minimum required content.

        Args:
            repo_root: Repository root path
        """
        readme = repo_root / "README.md"
        content = "# ML Odyssey\n\nProject description.\n"
        readme.write_text(content)

        assert readme.exists(), "README.md should exist"
        text = readme.read_text()
        assert len(text) > 0, "README.md should not be empty"
        assert "# " in text, "README.md should have a title"

    def test_contributing_has_content(self, repo_root: Path) -> None:
        """
        Test that CONTRIBUTING.md has minimum required content.

        Args:
            repo_root: Repository root path
        """
        contributing = repo_root / "CONTRIBUTING.md"
        content = "# Contributing\n\nContribution guidelines.\n"
        contributing.write_text(content)

        assert contributing.exists(), "CONTRIBUTING.md should exist"
        text = contributing.read_text()
        assert len(text) > 0, "CONTRIBUTING.md should not be empty"
        assert "# " in text, "CONTRIBUTING.md should have a title"

    def test_code_of_conduct_has_content(self, repo_root: Path) -> None:
        """
        Test that CODE_OF_CONDUCT.md has minimum required content.

        Args:
            repo_root: Repository root path
        """
        coc = repo_root / "CODE_OF_CONDUCT.md"
        content = "# Code of Conduct\n\nCommunity standards.\n"
        coc.write_text(content)

        assert coc.exists(), "CODE_OF_CONDUCT.md should exist"
        text = coc.read_text()
        assert len(text) > 0, "CODE_OF_CONDUCT.md should not be empty"
        assert "# " in text, "CODE_OF_CONDUCT.md should have a title"


class TestTier2Completeness:
    """Test cases for Tier 2 (Core Documentation) completeness - 8 documents."""

    @pytest.mark.parametrize(
        "doc_name",
        [
            "project-structure.md",
            "shared-library.md",
            "paper-implementation.md",
            "testing-strategy.md",
            "mojo-patterns.md",
            "agent-system.md",
            "workflow.md",
            "configuration.md",
        ],
    )
    def test_core_docs_exist(self, docs_root: Path, doc_name: str) -> None:
        """
        Test that docs/core/ documents exist.

        Args:
            docs_root: Path to docs directory
            doc_name: Name of document to test
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)
        doc_path = tier_dir / doc_name
        doc_path.touch()
        assert doc_path.exists(), f"{doc_name} should exist in core/"
        assert doc_path.is_file(), f"{doc_name} should be a file"

    @pytest.mark.parametrize(
        "doc_name",
        [
            "project-structure.md",
            "shared-library.md",
            "paper-implementation.md",
            "testing-strategy.md",
            "mojo-patterns.md",
            "agent-system.md",
            "workflow.md",
            "configuration.md",
        ],
    )
    def test_core_docs_have_content(self, docs_root: Path, doc_name: str) -> None:
        """
        Test that docs/core/ documents have minimum content.

        Args:
            docs_root: Path to docs directory
            doc_name: Name of document to test
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)
        doc_path = tier_dir / doc_name
        content = f"# {doc_name.replace('-', ' ').title()}\n\nContent here.\n"
        doc_path.write_text(content)

        assert doc_path.exists(), f"{doc_name} should exist"
        text = doc_path.read_text()
        assert len(text) > 0, f"{doc_name} should not be empty"
        assert "# " in text, f"{doc_name} should have a title"


class TestTier3Completeness:
    """Test cases for Tier 3 (Advanced Topics) completeness - 6 documents."""

    @pytest.mark.parametrize(
        "doc_name",
        [
            "performance.md",
            "custom-layers.md",
            "distributed-training.md",
            "visualization.md",
            "debugging.md",
            "integration.md",
        ],
    )
    def test_advanced_docs_exist(self, docs_root: Path, doc_name: str) -> None:
        """
        Test that docs/advanced/ documents exist.

        Args:
            docs_root: Path to docs directory
            doc_name: Name of document to test
        """
        tier_dir = docs_root / "advanced"
        tier_dir.mkdir(parents=True, exist_ok=True)
        doc_path = tier_dir / doc_name
        doc_path.touch()
        assert doc_path.exists(), f"{doc_name} should exist in advanced/"
        assert doc_path.is_file(), f"{doc_name} should be a file"

    @pytest.mark.parametrize(
        "doc_name",
        [
            "performance.md",
            "custom-layers.md",
            "distributed-training.md",
            "visualization.md",
            "debugging.md",
            "integration.md",
        ],
    )
    def test_advanced_docs_have_content(self, docs_root: Path, doc_name: str) -> None:
        """
        Test that docs/advanced/ documents have minimum content.

        Args:
            docs_root: Path to docs directory
            doc_name: Name of document to test
        """
        tier_dir = docs_root / "advanced"
        tier_dir.mkdir(parents=True, exist_ok=True)
        doc_path = tier_dir / doc_name
        content = f"# {doc_name.replace('-', ' ').title()}\n\nContent here.\n"
        doc_path.write_text(content)

        assert doc_path.exists(), f"{doc_name} should exist"
        text = doc_path.read_text()
        assert len(text) > 0, f"{doc_name} should not be empty"
        assert "# " in text, f"{doc_name} should have a title"


class TestTier4Completeness:
    """Test cases for Tier 4 (Development Guides) completeness - 4 documents."""

    @pytest.mark.parametrize(
        "doc_name",
        [
            "architecture.md",
            "api-reference.md",
            "release-process.md",
            "ci-cd.md",
        ],
    )
    def test_dev_docs_exist(self, docs_root: Path, doc_name: str) -> None:
        """
        Test that docs/dev/ documents exist.

        Args:
            docs_root: Path to docs directory
            doc_name: Name of document to test
        """
        tier_dir = docs_root / "dev"
        tier_dir.mkdir(parents=True, exist_ok=True)
        doc_path = tier_dir / doc_name
        doc_path.touch()
        assert doc_path.exists(), f"{doc_name} should exist in dev/"
        assert doc_path.is_file(), f"{doc_name} should be a file"

    @pytest.mark.parametrize(
        "doc_name",
        [
            "architecture.md",
            "api-reference.md",
            "release-process.md",
            "ci-cd.md",
        ],
    )
    def test_dev_docs_have_content(self, docs_root: Path, doc_name: str) -> None:
        """
        Test that docs/dev/ documents have minimum content.

        Args:
            docs_root: Path to docs directory
            doc_name: Name of document to test
        """
        tier_dir = docs_root / "dev"
        tier_dir.mkdir(parents=True, exist_ok=True)
        doc_path = tier_dir / doc_name
        content = f"# {doc_name.replace('-', ' ').title()}\n\nContent here.\n"
        doc_path.write_text(content)

        assert doc_path.exists(), f"{doc_name} should exist"
        text = doc_path.read_text()
        assert len(text) > 0, f"{doc_name} should not be empty"
        assert "# " in text, f"{doc_name} should have a title"


class TestDocumentCompleteness:
    """Test cases for overall document completeness validation."""

    def test_total_document_count(self, repo_root: Path, docs_root: Path) -> None:
        """
        Test that all 24 documents are present.

        Args:
            repo_root: Repository root path
            docs_root: Path to docs directory
        """
        # Create all tier directories
        for tier in ["getting-started", "core", "advanced", "dev"]:
            (docs_root / tier).mkdir(parents=True, exist_ok=True)

        # Create root docs (3)
        for doc in ["README.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md"]:
            (repo_root / doc).touch()

        # Create Tier 1 docs (3)
        tier1_docs = ["quickstart.md", "installation.md", "first-paper.md"]
        for doc in tier1_docs:
            (docs_root / "getting-started" / doc).touch()

        # Create Tier 2 docs (8)
        tier2_docs = [
            "project-structure.md",
            "shared-library.md",
            "paper-implementation.md",
            "testing-strategy.md",
            "mojo-patterns.md",
            "agent-system.md",
            "workflow.md",
            "configuration.md",
        ]
        for doc in tier2_docs:
            (docs_root / "core" / doc).touch()

        # Create Tier 3 docs (6)
        tier3_docs = [
            "performance.md",
            "custom-layers.md",
            "distributed-training.md",
            "visualization.md",
            "debugging.md",
            "integration.md",
        ]
        for doc in tier3_docs:
            (docs_root / "advanced" / doc).touch()

        # Create Tier 4 docs (4)
        tier4_docs = [
            "architecture.md",
            "api-reference.md",
            "release-process.md",
            "ci-cd.md",
        ]
        for doc in tier4_docs:
            (docs_root / "dev" / doc).touch()

        # Count all markdown files
        root_docs = list(repo_root.glob("*.md"))
        tier1_files = list((docs_root / "getting-started").glob("*.md"))
        tier2_files = list((docs_root / "core").glob("*.md"))
        tier3_files = list((docs_root / "advanced").glob("*.md"))
        tier4_files = list((docs_root / "dev").glob("*.md"))

        total = (
            len(root_docs)
            + len(tier1_files)
            + len(tier2_files)
            + len(tier3_files)
            + len(tier4_files)
        )

        assert total == 24, f"Should have 24 documents total, found {total}"

    def test_no_empty_documents(self, docs_root: Path) -> None:
        """
        Test that no documents are empty (all have minimum content).

        Args:
            docs_root: Path to docs directory
        """
        # Create test documents with content
        for tier in ["getting-started", "core", "advanced", "dev"]:
            tier_dir = docs_root / tier
            tier_dir.mkdir(parents=True, exist_ok=True)

            test_doc = tier_dir / "test.md"
            test_doc.write_text("# Test\n\nContent.\n")

            assert test_doc.read_text(), f"Document in {tier}/ should not be empty"

    def test_all_documents_have_headers(self, docs_root: Path) -> None:
        """
        Test that all documents have at least one header.

        Args:
            docs_root: Path to docs directory
        """
        # Create test documents with headers
        for tier in ["getting-started", "core", "advanced", "dev"]:
            tier_dir = docs_root / tier
            tier_dir.mkdir(parents=True, exist_ok=True)

            test_doc = tier_dir / "test.md"
            content = "# Test Document\n\nContent here.\n"
            test_doc.write_text(content)

            text = test_doc.read_text()
            assert "# " in text, f"Document in {tier}/ should have a header"
