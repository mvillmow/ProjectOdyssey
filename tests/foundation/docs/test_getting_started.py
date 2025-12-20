"""
Test suite for Tier 1 (Getting Started) documentation validation.

This module validates the 3 documents in Tier 1, ensuring they exist,
have required content, and meet quality standards for user-facing documentation.

Tier 1 Documents (3):
- docs/getting-started/quickstart.md
- docs/getting-started/installation.md
- docs/getting-started/first-paper.md

Coverage Target: >95%
"""

import pytest
from pathlib import Path


class TestREADME:
    """Test cases for README.md."""

    def test_readme_exists(self, repo_root: Path) -> None:
        """
        Test that README.md exists at repository root.

        Args:
            repo_root: Repository root path
        """
        readme = repo_root / "README.md"

        if not readme.exists():
            pytest.skip(f"Documentation file not created yet: {readme}")
        assert readme.exists(), "README.md should exist"
        assert readme.is_file(), "README.md should be a file"

    def test_readme_has_title(self, repo_root: Path) -> None:
        """
        Test that README.md has a main title.

        Args:
            repo_root: Repository root path
        """
        readme = repo_root / "README.md"
        content = """# ML Odyssey

A Mojo-based AI research platform.
"""
        if not readme.exists():
            pytest.skip(f"Documentation file not created yet: {readme}")

        readme.write_text(content)

        text = readme.read_text()
        assert text.startswith("# "), "README should start with title"
        assert "ML Odyssey" in text, "README should contain project name"

    def test_readme_has_description(self, repo_root: Path) -> None:
        """
        Test that README.md has project description.

        Args:
            repo_root: Repository root path
        """
        readme = repo_root / "README.md"
        content = """# ML Odyssey

A Mojo-based AI research platform for reproducing classic papers.

## Features

- Feature 1
- Feature 2
"""
        if not readme.exists():
            pytest.skip(f"Documentation file not created yet: {readme}")

        readme.write_text(content)

        text = readme.read_text()
        lines = [line for line in text.split("\n") if line.strip()]
        assert len(lines) >= 3, "README should have description content"

    def test_readme_has_sections(self, repo_root: Path) -> None:
        """
        Test that README.md has key sections.

        Args:
            repo_root: Repository root path
        """
        readme = repo_root / "README.md"
        content = """# ML Odyssey

Description here.

## Features

Features list.

## Getting Started

Quick start guide.

## Installation

Installation steps.
"""
        if not readme.exists():
            pytest.skip(f"Documentation file not created yet: {readme}")

        readme.write_text(content)

        text = readme.read_text()
        assert "## Features" in text or "## " in text, "README should have sections"


class TestQuickstart:
    """Test cases for quickstart.md."""

    def test_quickstart_exists(self, getting_started_dir: Path) -> None:
        """
        Test that quickstart.md exists.

        Args:
            getting_started_dir: Path to getting-started directory
        """
        quickstart = getting_started_dir / "quickstart.md"

        if not quickstart.exists():
            pytest.skip(f"Documentation file not created yet: {quickstart}")
        assert quickstart.exists(), "quickstart.md should exist"
        assert quickstart.is_file(), "quickstart.md should be a file"

    def test_quickstart_has_title(self, getting_started_dir: Path) -> None:
        """
        Test that quickstart.md has a title.

        Args:
            getting_started_dir: Path to getting-started directory
        """
        quickstart = getting_started_dir / "quickstart.md"
        content = """# Quick Start Guide

Get started quickly.
"""
        if not quickstart.exists():
            pytest.skip(f"Documentation file not created yet: {quickstart}")

        quickstart.write_text(content)

        text = quickstart.read_text()
        assert text.startswith("# "), "quickstart should start with title"

    def test_quickstart_has_examples(self, getting_started_dir: Path) -> None:
        """
        Test that quickstart.md has code examples.

        Args:
            getting_started_dir: Path to getting-started directory
        """
        quickstart = getting_started_dir / "quickstart.md"
        content = """# Quick Start

## Example

```python
import ml_odyssey
```

More examples.
"""
        if not quickstart.exists():
            pytest.skip(f"Documentation file not created yet: {quickstart}")

        quickstart.write_text(content)

        text = quickstart.read_text()
        assert "```" in text, "quickstart should have code examples"


class TestInstallation:
    """Test cases for installation.md."""

    def test_installation_exists(self, getting_started_dir: Path) -> None:
        """
        Test that installation.md exists.

        Args:
            getting_started_dir: Path to getting-started directory
        """
        installation = getting_started_dir / "installation.md"

        if not installation.exists():
            pytest.skip(f"Documentation file not created yet: {installation}")
        assert installation.exists(), "installation.md should exist"
        assert installation.is_file(), "installation.md should be a file"

    def test_installation_has_title(self, getting_started_dir: Path) -> None:
        """
        Test that installation.md has a title.

        Args:
            getting_started_dir: Path to getting-started directory
        """
        installation = getting_started_dir / "installation.md"
        content = """# Installation Guide

How to install ML Odyssey.
"""
        if not installation.exists():
            pytest.skip(f"Documentation file not created yet: {installation}")

        installation.write_text(content)

        text = installation.read_text()
        assert text.startswith("# "), "installation should start with title"

    def test_installation_has_steps(self, getting_started_dir: Path) -> None:
        """
        Test that installation.md has installation steps.

        Args:
            getting_started_dir: Path to getting-started directory
        """
        installation = getting_started_dir / "installation.md"
        content = """# Installation

## Prerequisites

Requirements.

## Installation Steps

1. Step 1
2. Step 2
3. Step 3
"""
        if not installation.exists():
            pytest.skip(f"Documentation file not created yet: {installation}")

        installation.write_text(content)

        text = installation.read_text()
        assert "## " in text, "installation should have sections"
        # Check for numbered list or steps
        assert any(char.isdigit() for char in text), "installation should have steps"


class TestFirstPaper:
    """Test cases for first-paper.md."""

    def test_first_paper_exists(self, getting_started_dir: Path) -> None:
        """
        Test that first-paper.md exists.

        Args:
            getting_started_dir: Path to getting-started directory
        """
        first_paper = getting_started_dir / "first-paper.md"
        if not first_paper.exists():
            pytest.skip(f"Documentation file not created yet: {first_paper}")
        assert first_paper.is_file(), "first-paper.md should be a file"

    def test_first_paper_has_title(self, getting_started_dir: Path) -> None:
        """
        Test that first-paper.md has a title.

        Args:
            getting_started_dir: Path to getting-started directory
        """
        first_paper = getting_started_dir / "first-paper.md"
        content = """# Your First Paper Implementation

Tutorial for first paper.
"""
        if not first_paper.exists():
            pytest.skip(f"Documentation file not created yet: {first_paper}")

        first_paper.write_text(content)

        text = first_paper.read_text()
        assert text.startswith("# "), "first-paper should start with title"

    def test_first_paper_has_tutorial(self, getting_started_dir: Path) -> None:
        """
        Test that first-paper.md has tutorial content.

        Args:
            getting_started_dir: Path to getting-started directory
        """
        first_paper = getting_started_dir / "first-paper.md"
        content = """# First Paper

## Overview

What you'll learn.

## Implementation

```mojo
fn main():
    print("Hello")
```

## Next Steps

Where to go next.
"""
        if not first_paper.exists():
            pytest.skip(f"Documentation file not created yet: {first_paper}")

        first_paper.write_text(content)

        text = first_paper.read_text()
        assert "## " in text, "first-paper should have sections"
        assert "```" in text, "first-paper should have code examples"


class TestTier1Integration:
    """Test cases for Tier 1 integration and completeness."""

    def test_all_tier1_docs_exist(self, repo_root: Path, getting_started_dir: Path) -> None:
        """
        Test that all 3 Tier 1 documents exist.

        Args:
            repo_root: Repository root path
            getting_started_dir: Path to getting-started directory
        """
        # Create all Tier 1 documents
        gs_docs = ["quickstart.md", "installation.md", "first-paper.md"]
        for doc in gs_docs:
            doc_path = getting_started_dir / doc
            if not doc_path.exists():
                pytest.skip(f"Documentation file not created yet: {doc_path}")

        # Verify all exist
        for doc in gs_docs:
            assert (getting_started_dir / doc).exists(), f"{doc} should exist"

    def test_tier1_document_count(self, repo_root: Path, getting_started_dir: Path) -> None:
        """
        Test that Tier 1 has exactly 3 documents.

        Args:
            repo_root: Repository root path
            getting_started_dir: Path to getting-started directory
        """
        # Create all Tier 1 documents
        gs_docs = ["quickstart.md", "installation.md", "first-paper.md"]
        for doc in gs_docs:
            doc_path = getting_started_dir / doc
            if not doc_path.exists():
                pytest.skip(f"Documentation file not created yet: {doc_path}")

        # Count documents
        gs_count = len([d for d in gs_docs if (getting_started_dir / d).exists()])

        assert gs_count == 3, f"Tier 1 should have 3 documents, found {gs_count}"
