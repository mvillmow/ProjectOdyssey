"""
Test suite for tools/ directory documentation completeness.

This module validates that README.md files exist and contain required sections
explaining purpose, language strategy, and contribution guidelines.

Test Categories:
- Main README validation: tools/README.md completeness
- Category README validation: Each category has documentation
- Documentation structure: Required sections present

Coverage Target: 95% (focus on critical documentation)
"""

from pathlib import Path
from typing import Dict

import pytest


class TestMainReadmeDocumentation:
    """Test cases for main tools/README.md documentation."""

    def test_main_readme_exists(self, main_readme: Path) -> None:
        """
        Test that main tools/README.md file exists.

        Verifies:
        - README.md exists in tools/ directory
        - File is not empty

        Args:
            main_readme: Path to tools/README.md
        """
        assert main_readme.exists(), "tools/README.md should exist"
        assert main_readme.is_file(), "tools/README.md should be a file"

        # Verify file is not empty
        content = main_readme.read_text()
        assert len(content) > 0, "tools/README.md should not be empty"

    def test_main_readme_has_purpose(self, main_readme: Path) -> None:
        """
        Test that main README contains purpose or overview section.

        Verifies:
        - README explains what tools/ directory is for
        - Contains "Purpose", "Overview", or similar heading

        Args:
            main_readme: Path to tools/README.md
        """
        content = main_readme.read_text().lower()

        # Check for purpose/overview section
        has_purpose = any(
            keyword in content
            for keyword in ["## purpose", "## overview", "# tools"]
        )

        assert has_purpose, (
            "tools/README.md should have a Purpose or Overview section"
        )

    def test_main_readme_has_categories(
        self, main_readme: Path, category_names: list
    ) -> None:
        """
        Test that main README documents all 4 tool categories.

        Verifies:
        - README mentions paper-scaffold category
        - README mentions test-utils category
        - README mentions benchmarking category
        - README mentions codegen category

        Args:
            main_readme: Path to tools/README.md
            category_names: List of expected category names
        """
        content = main_readme.read_text().lower()

        # Check each category is mentioned
        for category in category_names:
            # Convert hyphenated names for matching (paper-scaffold -> paper scaffold)
            category_variations = [
                category,
                category.replace("-", " "),
                category.replace("-", ""),
            ]

            category_mentioned = any(
                variation in content for variation in category_variations
            )

            assert category_mentioned, (
                f"tools/README.md should document {category} category"
            )

    def test_main_readme_has_language_strategy(self, main_readme: Path) -> None:
        """
        Test that main README documents language selection strategy.

        Verifies:
        - README mentions Mojo and Python language choices
        - README references or explains language selection

        Args:
            main_readme: Path to tools/README.md
        """
        content = main_readme.read_text().lower()

        # Check for language strategy mentions
        has_mojo = "mojo" in content
        has_python = "python" in content
        has_language = "language" in content

        assert has_mojo, "tools/README.md should mention Mojo"
        assert has_python, "tools/README.md should mention Python"
        assert has_language, (
            "tools/README.md should discuss language selection strategy"
        )

    def test_main_readme_has_contribution_guide(
        self, main_readme: Path
    ) -> None:
        """
        Test that main README has contribution guidelines.

        Verifies:
        - README contains "Contributing" or similar section
        - README provides guidance on adding new tools

        Args:
            main_readme: Path to tools/README.md
        """
        content = main_readme.read_text().lower()

        # Check for contribution section
        has_contributing = any(
            keyword in content
            for keyword in [
                "## contributing",
                "## contribution",
                "adding a new tool",
                "adding new tools",
            ]
        )

        assert has_contributing, (
            "tools/README.md should have contribution guidelines"
        )

    def test_adr_001_reference(self, main_readme: Path) -> None:
        """
        Test that main README references ADR-001 for language selection.

        Verifies:
        - README links to or mentions ADR-001

        Args:
            main_readme: Path to tools/README.md
        """
        content = main_readme.read_text()

        # Check for ADR-001 reference
        has_adr_reference = "ADR-001" in content or "adr-001" in content.lower()

        assert has_adr_reference, (
            "tools/README.md should reference ADR-001 for language selection"
        )

    def test_main_readme_has_structure_diagram(self, main_readme: Path) -> None:
        """
        Test that main README shows directory structure.

        Verifies:
        - README contains directory structure diagram or example

        Args:
            main_readme: Path to tools/README.md
        """
        content = main_readme.read_text().lower()

        # Check for structure indicators
        has_structure = any(
            keyword in content
            for keyword in [
                "```",  # Code block (likely showing structure)
                "directory structure",
                "structure",
                "tools/",
            ]
        )

        assert has_structure, (
            "tools/README.md should show directory structure"
        )


class TestCategoryReadmeDocumentation:
    """Test cases for category README.md documentation."""

    def test_category_readmes_exist(
        self, category_readmes: Dict[str, Path]
    ) -> None:
        """
        Test that all category directories have README.md files.

        Verifies:
        - Each category has a README.md
        - README files are not empty

        Args:
            category_readmes: Dictionary of category README paths
        """
        for category_name, readme_path in category_readmes.items():
            assert readme_path.exists(), (
                f"tools/{category_name}/README.md should exist"
            )
            assert readme_path.is_file(), (
                f"tools/{category_name}/README.md should be a file"
            )

            # Verify file is not empty
            content = readme_path.read_text()
            assert len(content) > 0, (
                f"tools/{category_name}/README.md should not be empty"
            )

    def test_category_readme_has_title(
        self, category_readmes: Dict[str, Path]
    ) -> None:
        """
        Test that each category README has a title.

        Verifies:
        - README starts with # heading
        - Heading relates to category name

        Args:
            category_readmes: Dictionary of category README paths
        """
        for category_name, readme_path in category_readmes.items():
            content = readme_path.read_text()
            lines = content.split("\n")

            # Find first non-empty line
            first_line = ""
            for line in lines:
                if line.strip():
                    first_line = line.strip()
                    break

            assert first_line.startswith("#"), (
                f"{category_name}/README.md should start with a heading"
            )

    def test_category_readme_has_coming_soon_or_content(
        self, category_readmes: Dict[str, Path]
    ) -> None:
        """
        Test that category READMEs have content or "Coming Soon" placeholder.

        Verifies:
        - README has either tool documentation or "Coming Soon" section
        - README indicates future plans if no tools exist yet

        Args:
            category_readmes: Dictionary of category README paths
        """
        for category_name, readme_path in category_readmes.items():
            content = readme_path.read_text().lower()

            # Check for either substantial content or "Coming Soon" placeholder
            has_content = len(content) > 100  # Arbitrary threshold
            has_coming_soon = "coming soon" in content

            assert has_content or has_coming_soon, (
                f"{category_name}/README.md should have content or "
                "'Coming Soon' placeholder"
            )

    def test_paper_scaffold_readme_describes_purpose(
        self, category_readmes: Dict[str, Path]
    ) -> None:
        """
        Test that paper-scaffold README describes its purpose.

        Verifies:
        - README explains scaffolding functionality
        - README mentions templates or paper structure

        Args:
            category_readmes: Dictionary of category README paths
        """
        readme_path = category_readmes["paper-scaffold"]
        content = readme_path.read_text().lower()

        # Check for scaffolding-related content
        has_purpose = any(
            keyword in content
            for keyword in [
                "scaffold",
                "template",
                "paper",
                "structure",
                "generate",
            ]
        )

        assert has_purpose, (
            "paper-scaffold/README.md should describe scaffolding purpose"
        )

    def test_test_utils_readme_describes_purpose(
        self, category_readmes: Dict[str, Path]
    ) -> None:
        """
        Test that test-utils README describes its purpose.

        Verifies:
        - README explains testing utilities
        - README mentions test data or fixtures

        Args:
            category_readmes: Dictionary of category README paths
        """
        readme_path = category_readmes["test-utils"]
        content = readme_path.read_text().lower()

        # Check for testing-related content
        has_purpose = any(
            keyword in content
            for keyword in ["test", "fixture", "data", "utility", "utilities"]
        )

        assert has_purpose, (
            "test-utils/README.md should describe testing utilities purpose"
        )

    def test_benchmarking_readme_describes_purpose(
        self, category_readmes: Dict[str, Path]
    ) -> None:
        """
        Test that benchmarking README describes its purpose.

        Verifies:
        - README explains benchmarking functionality
        - README mentions performance or measurement

        Args:
            category_readmes: Dictionary of category README paths
        """
        readme_path = category_readmes["benchmarking"]
        content = readme_path.read_text().lower()

        # Check for benchmarking-related content
        has_purpose = any(
            keyword in content
            for keyword in [
                "benchmark",
                "performance",
                "measure",
                "speed",
                "timing",
            ]
        )

        assert has_purpose, (
            "benchmarking/README.md should describe benchmarking purpose"
        )

    def test_codegen_readme_describes_purpose(
        self, category_readmes: Dict[str, Path]
    ) -> None:
        """
        Test that codegen README describes its purpose.

        Verifies:
        - README explains code generation functionality
        - README mentions templates or boilerplate

        Args:
            category_readmes: Dictionary of category README paths
        """
        readme_path = category_readmes["codegen"]
        content = readme_path.read_text().lower()

        # Check for code generation-related content
        has_purpose = any(
            keyword in content
            for keyword in [
                "code generation",
                "codegen",
                "generate",
                "template",
                "boilerplate",
            ]
        )

        assert has_purpose, (
            "codegen/README.md should describe code generation purpose"
        )


class TestDocumentationQuality:
    """Test cases for overall documentation quality."""

    def test_all_readmes_use_markdown(
        self, main_readme: Path, category_readmes: Dict[str, Path]
    ) -> None:
        """
        Test that all README files use markdown format.

        Verifies:
        - Files are named README.md (not README.txt)
        - Files contain markdown syntax

        Args:
            main_readme: Path to main README
            category_readmes: Dictionary of category README paths
        """
        # Check main README
        assert main_readme.name == "README.md", (
            "Main README should be named README.md"
        )

        # Check category READMEs
        for category_name, readme_path in category_readmes.items():
            assert readme_path.name == "README.md", (
                f"{category_name} README should be named README.md"
            )

            # Verify markdown syntax present
            content = readme_path.read_text()
            has_markdown = "#" in content or "```" in content or "[" in content

            assert has_markdown, (
                f"{category_name}/README.md should use markdown syntax"
            )

    def test_readmes_have_reasonable_length(
        self, main_readme: Path, category_readmes: Dict[str, Path]
    ) -> None:
        """
        Test that README files have reasonable length (not too short).

        Verifies:
        - Main README has substantial content (> 500 chars)
        - Category READMEs have minimal content (> 50 chars)

        Args:
            main_readme: Path to main README
            category_readmes: Dictionary of category README paths
        """
        # Main README should be substantial
        main_content = main_readme.read_text()
        assert len(main_content) > 500, (
            "tools/README.md should have substantial content (> 500 characters)"
        )

        # Category READMEs should have at least minimal content
        for category_name, readme_path in category_readmes.items():
            content = readme_path.read_text()
            assert len(content) > 50, (
                f"{category_name}/README.md should have content (> 50 characters)"
            )
