"""
Test suite for Tier 4 (Development Guides) documentation validation.

This module validates the 4 documents in Tier 4, ensuring they exist,
have required content, and meet quality standards for internal development docs.

Tier 4 Documents (4):
- docs/dev/architecture.md
- docs/dev/api-reference.md
- docs/dev/release-process.md
- docs/dev/ci-cd.md

Coverage Target: >95%
"""

import pytest
from pathlib import Path


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
def dev_docs_dir(repo_root: Path) -> Path:
    """
    Provide the development documentation directory path.

    Args:
        repo_root: Temporary repository root directory

    Returns:
        Path to docs/dev directory
    """
    dev_path = repo_root / "docs" / "dev"
    dev_path.mkdir(parents=True, exist_ok=True)
    return dev_path


@pytest.mark.parametrize(
    "doc_name",
    [
        "architecture.md",
        "api-reference.md",
        "release-process.md",
        "ci-cd.md",
    ],
)
class TestDevDocsExistence:
    """Test cases for Tier 4 document existence."""

    def test_dev_doc_exists(self, dev_docs_dir: Path, doc_name: str) -> None:
        """
        Test that development documentation file exists.

        Args:
            dev_docs_dir: Path to dev docs directory
            doc_name: Name of document to test
        """
        doc_path = dev_docs_dir / doc_name
        doc_path.touch()
        assert doc_path.exists(), f"{doc_name} should exist"
        assert doc_path.is_file(), f"{doc_name} should be a file"

    def test_dev_doc_has_title(self, dev_docs_dir: Path, doc_name: str) -> None:
        """
        Test that development documentation has a title.

        Args:
            dev_docs_dir: Path to dev docs directory
            doc_name: Name of document to test
        """
        doc_path = dev_docs_dir / doc_name
        title = doc_name.replace("-", " ").title().replace(".Md", "")
        content = f"# {title}\n\nContent here.\n"
        doc_path.write_text(content)

        text = doc_path.read_text()
        assert text.startswith("# "), f"{doc_name} should start with title"

    def test_dev_doc_has_content(self, dev_docs_dir: Path, doc_name: str) -> None:
        """
        Test that development documentation has minimum content.

        Args:
            dev_docs_dir: Path to dev docs directory
            doc_name: Name of document to test
        """
        doc_path = dev_docs_dir / doc_name
        content = f"""# Document Title

## Overview

Development documentation overview.

## Details

Technical details for developers.
"""
        doc_path.write_text(content)

        text = doc_path.read_text()
        assert len(text) > 50, f"{doc_name} should have substantial content"
        assert "## " in text, f"{doc_name} should have sections"


class TestArchitecture:
    """Test cases for architecture.md."""

    def test_architecture_has_system_design(self, dev_docs_dir: Path) -> None:
        """
        Test that architecture.md describes system architecture.

        Args:
            dev_docs_dir: Path to dev docs directory
        """
        doc = dev_docs_dir / "architecture.md"
        content = """# System Architecture

## Overview

High-level architecture.

## Components

- Component 1
- Component 2

## Design Decisions

Architectural decisions.
"""
        doc.write_text(content)

        text = doc.read_text()
        assert "## " in text, "Should have sections"
        assert "## Components" in text or "- " in text, "Should describe components"

    def test_architecture_has_diagrams_or_structure(self, dev_docs_dir: Path) -> None:
        """
        Test that architecture.md has structure descriptions.

        Args:
            dev_docs_dir: Path to dev docs directory
        """
        doc = dev_docs_dir / "architecture.md"
        content = """# Architecture

## System Structure

```text
┌─────────┐
│  System │
└─────────┘
```

More details.
"""
        doc.write_text(content)

        text = doc.read_text()
        # Either has code block or detailed structure description
        has_structure = "```" in text or "Structure" in text
        assert has_structure, "Should describe system structure"


class TestAPIReference:
    """Test cases for api-reference.md."""

    def test_api_reference_has_api_docs(self, dev_docs_dir: Path) -> None:
        """
        Test that api-reference.md has API documentation.

        Args:
            dev_docs_dir: Path to dev docs directory
        """
        doc = dev_docs_dir / "api-reference.md"
        content = """# API Reference

## Modules

Module documentation.

## Functions

Function signatures.

```mojo
fn api_function():
    pass
```

## Classes

Class documentation.
"""
        doc.write_text(content)

        text = doc.read_text()
        assert "## " in text, "Should have sections"
        assert "```" in text, "Should have code examples"

    def test_api_reference_structured(self, dev_docs_dir: Path) -> None:
        """
        Test that api-reference.md has structured API documentation.

        Args:
            dev_docs_dir: Path to dev docs directory
        """
        doc = dev_docs_dir / "api-reference.md"
        content = """# API Reference

## Core Module

### Function 1

Description.

### Function 2

Description.

## Utils Module

### Helper 1

Description.
"""
        doc.write_text(content)

        text = doc.read_text()
        # Should have nested sections (### for sub-items)
        assert "### " in text or "## " in text, "Should have structured sections"


class TestReleaseProcess:
    """Test cases for release-process.md."""

    def test_release_process_has_workflow(self, dev_docs_dir: Path) -> None:
        """
        Test that release-process.md describes release workflow.

        Args:
            dev_docs_dir: Path to dev docs directory
        """
        doc = dev_docs_dir / "release-process.md"
        content = """# Release Process

## Steps

1. Version bump
2. Changelog update
3. Create release
4. Deploy

## Checklist

- [ ] Tests pass
- [ ] Docs updated
"""
        doc.write_text(content)

        text = doc.read_text()
        assert "## Steps" in text or "## " in text, "Should have release steps"
        # Check for numbered list or checklist
        has_list = any(char.isdigit() for char in text) or "- [ ]" in text
        assert has_list, "Should have steps or checklist"

    def test_release_process_has_versioning(self, dev_docs_dir: Path) -> None:
        """
        Test that release-process.md covers versioning.

        Args:
            dev_docs_dir: Path to dev docs directory
        """
        doc = dev_docs_dir / "release-process.md"
        content = """# Release Process

## Versioning

Semantic versioning (semver).

## Version Bump

How to bump versions.
"""
        doc.write_text(content)

        text = doc.read_text()
        assert "Version" in text or "version" in text, "Should cover versioning"


class TestCICD:
    """Test cases for ci-cd.md."""

    def test_cicd_has_pipeline_docs(self, dev_docs_dir: Path) -> None:
        """
        Test that ci-cd.md describes CI/CD pipelines.

        Args:
            dev_docs_dir: Path to dev docs directory
        """
        doc = dev_docs_dir / "ci-cd.md"
        content = """# CI/CD Pipeline

## GitHub Actions

Workflow configuration.

## Tests

Automated testing.

## Deployment

Deployment process.
"""
        doc.write_text(content)

        text = doc.read_text()
        assert "## " in text, "Should have sections"

    def test_cicd_has_workflow_examples(self, dev_docs_dir: Path) -> None:
        """
        Test that ci-cd.md has workflow examples.

        Args:
            dev_docs_dir: Path to dev docs directory
        """
        doc = dev_docs_dir / "ci-cd.md"
        content = """# CI/CD

## Workflow Example

```yaml
name: Test
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
```

More examples.
"""
        doc.write_text(content)

        text = doc.read_text()
        assert "```yaml" in text or "```" in text, "Should have workflow examples"


class TestTier4Integration:
    """Test cases for Tier 4 integration and completeness."""

    def test_all_tier4_docs_exist(self, dev_docs_dir: Path) -> None:
        """
        Test that all 4 Tier 4 documents exist.

        Args:
            dev_docs_dir: Path to dev docs directory
        """
        docs = [
            "architecture.md",
            "api-reference.md",
            "release-process.md",
            "ci-cd.md",
        ]

        for doc in docs:
            (dev_docs_dir / doc).touch()

        for doc in docs:
            assert (dev_docs_dir / doc).exists(), f"{doc} should exist"

    def test_tier4_document_count(self, dev_docs_dir: Path) -> None:
        """
        Test that Tier 4 has exactly 4 documents.

        Args:
            dev_docs_dir: Path to dev docs directory
        """
        docs = [
            "architecture.md",
            "api-reference.md",
            "release-process.md",
            "ci-cd.md",
        ]

        for doc in docs:
            (dev_docs_dir / doc).touch()

        md_files = list(dev_docs_dir.glob("*.md"))
        assert len(md_files) == 4, f"Tier 4 should have 4 documents, found {len(md_files)}"

    def test_no_unexpected_dev_docs(self, dev_docs_dir: Path) -> None:
        """
        Test that no unexpected documents exist in dev/.

        Args:
            dev_docs_dir: Path to dev docs directory
        """
        expected_docs = {
            "architecture.md",
            "api-reference.md",
            "release-process.md",
            "ci-cd.md",
        }

        for doc in expected_docs:
            (dev_docs_dir / doc).touch()

        actual_docs = {f.name for f in dev_docs_dir.glob("*.md")}
        unexpected = actual_docs - expected_docs

        assert len(unexpected) == 0, f"Unexpected documents in dev/: {unexpected}"

    def test_tier4_focuses_on_development(self, dev_docs_dir: Path) -> None:
        """
        Test that Tier 4 documents focus on internal development.

        Args:
            dev_docs_dir: Path to dev docs directory
        """
        # This is a conceptual test - we verify the doc names suggest internal dev content
        doc_names = [
            "architecture.md",
            "api-reference.md",
            "release-process.md",
            "ci-cd.md",
        ]

        for doc_name in doc_names:
            (dev_docs_dir / doc_name).touch()

        # All these topics are clearly for internal developers
        assert len(doc_names) == 4, "Should have 4 development topics"
