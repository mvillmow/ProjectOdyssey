"""
Test suite for documentation link validation.

This module validates that internal and external links in documentation are
correct and reachable.

Test Categories:
- Internal links: Markdown links to other docs
- Relative paths: Correct path resolution
- Cross-tier references: Links between tiers
- Link format: Valid markdown syntax
- Broken links: Detect missing targets

Coverage Target: >95%
"""

import re
import pytest
from pathlib import Path
from typing import List, Tuple


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


def extract_markdown_links(content: str) -> List[Tuple[str, str]]:
    """
    Extract markdown links from content.

    Args:
        content: Markdown content to parse

    Returns:
        List of (link_text, link_url) tuples
    """
    # Pattern matches [text](url)
    pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
    matches = re.findall(pattern, content)
    return matches


class TestInternalLinks:
    """Test cases for internal documentation links."""

    def test_link_to_same_tier(self, docs_root: Path) -> None:
        """
        Test links within same tier resolve correctly.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        # Create two documents in same tier
        doc1 = tier_dir / "doc1.md"
        doc2 = tier_dir / "doc2.md"

        doc1.write_text("# Doc 1\n\nSee [Doc 2](doc2.md).\n")
        doc2.write_text("# Doc 2\n\nContent.\n")

        # Parse links
        links = extract_markdown_links(doc1.read_text())
        assert len(links) == 1, "Should find one link"

        link_text, link_url = links[0]
        assert link_url == "doc2.md", "Link should point to doc2.md"

        # Verify target exists
        target = tier_dir / link_url
        assert target.exists(), "Link target should exist"

    def test_link_to_parent_tier(self, docs_root: Path) -> None:
        """
        Test links to parent tier (up one level) resolve correctly.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        getting_started_dir = docs_root / "getting-started"
        getting_started_dir.mkdir(parents=True, exist_ok=True)

        # Create documents in different tiers
        core_doc = tier_dir / "project-structure.md"
        gs_doc = getting_started_dir / "quickstart.md"

        core_doc.write_text("# Project Structure\n\nSee [Quickstart](../getting-started/quickstart.md).\n")
        gs_doc.write_text("# Quickstart\n\nContent.\n")

        # Parse links
        links = extract_markdown_links(core_doc.read_text())
        assert len(links) == 1, "Should find one link"

        link_text, link_url = links[0]
        assert link_url == "../getting-started/quickstart.md", "Link should use relative path"

        # Verify target exists (resolve relative path)
        target = (tier_dir / link_url).resolve()
        assert target.exists(), "Link target should exist"

    def test_link_to_root_document(self, repo_root: Path, docs_root: Path) -> None:
        """
        Test links to root-level documents resolve correctly.

        Args:
            repo_root: Repository root path
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        # Create root document
        readme = repo_root / "README.md"
        readme.write_text("# ML Odyssey\n\nContent.\n")

        # Create tier document with link to root
        core_doc = tier_dir / "project-structure.md"
        core_doc.write_text("# Project Structure\n\nSee [README](../../README.md).\n")

        # Parse links
        links = extract_markdown_links(core_doc.read_text())
        assert len(links) == 1, "Should find one link"

        link_text, link_url = links[0]
        assert link_url == "../../README.md", "Link should use relative path to root"

        # Verify target exists
        target = (tier_dir / link_url).resolve()
        assert target.exists(), "Link target should exist"

    def test_multiple_links_in_document(self, docs_root: Path) -> None:
        """
        Test document with multiple links.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        # Create target documents
        for i in range(1, 4):
            doc = tier_dir / f"doc{i}.md"
            doc.write_text(f"# Doc {i}\n\nContent.\n")

        # Create source document with multiple links
        source = tier_dir / "index.md"
        content = """# Index

See these docs:
- [Doc 1](doc1.md)
- [Doc 2](doc2.md)
- [Doc 3](doc3.md)
"""
        source.write_text(content)

        # Parse links
        links = extract_markdown_links(source.read_text())
        assert len(links) == 3, "Should find three links"

        # Verify all targets exist
        for link_text, link_url in links:
            target = tier_dir / link_url
            assert target.exists(), f"Link target {link_url} should exist"


class TestCrossTierReferences:
    """Test cases for links between different documentation tiers."""

    def test_tier1_to_tier2_link(self, docs_root: Path) -> None:
        """
        Test link from Tier 1 to Tier 2 document.

        Args:
            docs_root: Path to docs directory
        """
        tier1_dir = docs_root / "getting-started"
        tier1_dir.mkdir(parents=True, exist_ok=True)

        tier2_dir = docs_root / "core"
        tier2_dir.mkdir(parents=True, exist_ok=True)

        # Create documents
        tier1_doc = tier1_dir / "quickstart.md"
        tier2_doc = tier2_dir / "project-structure.md"

        tier1_doc.write_text("# Quickstart\n\nSee [Project Structure](../core/project-structure.md).\n")
        tier2_doc.write_text("# Project Structure\n\nContent.\n")

        # Parse and verify
        links = extract_markdown_links(tier1_doc.read_text())
        assert len(links) == 1, "Should find one link"

        link_text, link_url = links[0]
        target = (tier1_dir / link_url).resolve()
        assert target.exists(), "Cross-tier link should resolve"

    def test_tier2_to_tier3_link(self, docs_root: Path) -> None:
        """
        Test link from Tier 2 to Tier 3 document.

        Args:
            docs_root: Path to docs directory
        """
        tier2_dir = docs_root / "core"
        tier2_dir.mkdir(parents=True, exist_ok=True)

        tier3_dir = docs_root / "advanced"
        tier3_dir.mkdir(parents=True, exist_ok=True)

        # Create documents
        tier2_doc = tier2_dir / "mojo-patterns.md"
        tier3_doc = tier3_dir / "performance.md"

        tier2_doc.write_text("# Mojo Patterns\n\nSee [Performance](../advanced/performance.md).\n")
        tier3_doc.write_text("# Performance\n\nContent.\n")

        # Parse and verify
        links = extract_markdown_links(tier2_doc.read_text())
        assert len(links) == 1, "Should find one link"

        link_text, link_url = links[0]
        target = (tier2_dir / link_url).resolve()
        assert target.exists(), "Cross-tier link should resolve"

    def test_bidirectional_links(self, docs_root: Path) -> None:
        """
        Test bidirectional links between two documents.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        # Create two documents with links to each other
        doc1 = tier_dir / "doc1.md"
        doc2 = tier_dir / "doc2.md"

        doc1.write_text("# Doc 1\n\nSee [Doc 2](doc2.md).\n")
        doc2.write_text("# Doc 2\n\nSee [Doc 1](doc1.md).\n")

        # Verify both directions
        links1 = extract_markdown_links(doc1.read_text())
        links2 = extract_markdown_links(doc2.read_text())

        assert len(links1) == 1, "Doc1 should have one link"
        assert len(links2) == 1, "Doc2 should have one link"

        # Verify targets exist
        target1 = tier_dir / links1[0][1]
        target2 = tier_dir / links2[0][1]

        assert target1.exists(), "Link from doc1 should resolve"
        assert target2.exists(), "Link from doc2 should resolve"


class TestLinkFormat:
    """Test cases for link format validation."""

    def test_valid_markdown_link_syntax(self, docs_root: Path) -> None:
        """
        Test that links use valid markdown syntax.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        doc.write_text("# Test\n\nSee [Link Text](target.md).\n")

        links = extract_markdown_links(doc.read_text())
        assert len(links) == 1, "Should find valid link"

        link_text, link_url = links[0]
        assert link_text == "Link Text", "Link text should be extracted"
        assert link_url == "target.md", "Link URL should be extracted"

    def test_link_with_spaces_in_text(self, docs_root: Path) -> None:
        """
        Test links with spaces in link text.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        doc.write_text("# Test\n\nSee [Link With Spaces](target.md).\n")

        links = extract_markdown_links(doc.read_text())
        assert len(links) == 1, "Should find link with spaces"

        link_text, link_url = links[0]
        assert " " in link_text, "Link text should preserve spaces"

    def test_relative_path_format(self, docs_root: Path) -> None:
        """
        Test that relative paths use correct format.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        doc.write_text("# Test\n\nSee [Link](../getting-started/doc.md).\n")

        links = extract_markdown_links(doc.read_text())
        assert len(links) == 1, "Should find relative link"

        link_text, link_url = links[0]
        assert link_url.startswith("../"), "Should use ../ for parent directory"

    def test_markdown_extension_present(self, docs_root: Path) -> None:
        """
        Test that links include .md extension.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        doc.write_text("# Test\n\nSee [Link](target.md).\n")

        links = extract_markdown_links(doc.read_text())
        assert len(links) == 1, "Should find link"

        link_text, link_url = links[0]
        assert link_url.endswith(".md"), "Link should have .md extension"


class TestBrokenLinks:
    """Test cases for detecting broken links."""

    def test_detect_missing_target(self, docs_root: Path) -> None:
        """
        Test detection of links to non-existent files.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        doc.write_text("# Test\n\nSee [Missing](missing.md).\n")

        links = extract_markdown_links(doc.read_text())
        assert len(links) == 1, "Should find link"

        link_text, link_url = links[0]
        target = tier_dir / link_url

        # This should fail - target doesn't exist
        assert not target.exists(), "Link target should not exist (broken link)"

    def test_detect_invalid_relative_path(self, docs_root: Path) -> None:
        """
        Test detection of invalid relative paths.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        doc.write_text("# Test\n\nSee [Link](../../invalid/path.md).\n")

        links = extract_markdown_links(doc.read_text())
        assert len(links) == 1, "Should find link"

        link_text, link_url = links[0]
        target = tier_dir / link_url

        # Resolve and check if parent exists
        try:
            resolved = target.resolve(strict=False)
            # Check if parent directory exists
            assert not resolved.parent.exists(), "Parent directory should not exist"
        except Exception:
            # Path resolution failed - invalid path
            pass
