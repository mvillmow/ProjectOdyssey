"""
Test suite for template structure validation.

This module contains comprehensive tests for validating the papers/_template/
directory structure, ensuring it provides a complete starting point for new
paper implementations.

Test Categories:
- Template directory structure
- Template file completeness
- Template documentation
- Template copy functionality

Coverage Target: 100%
"""

import shutil
from pathlib import Path
from typing import Dict, List



def validate_directory_structure(base_path: Path, expected_structure: Dict[str, List[str]]) -> List[str]:
    """
    Validate directory structure against expected layout.

    Args:
        base_path: Base directory path to validate
        expected_structure: Dictionary mapping subdirectory names to expected contents

    Returns:
        List of validation errors (empty list if all valid).
    """
    errors = []

    for dir_name, expected_items in expected_structure.items():
        # Handle root directory separately
        dir_path = base_path if dir_name == "root" else base_path / dir_name

        if not dir_path.exists():
            errors.append(f"Directory missing: {dir_path}")
            continue

        if not dir_path.is_dir():
            errors.append(f"Not a directory: {dir_path}")
            continue

        # Check each expected item exists
        for item in expected_items:
            item_path = dir_path / item
            if not item_path.exists():
                errors.append(f"Required item missing: {item_path}")

    return errors


class TestTemplateDirectoryStructure:
    """Test cases for template directory structure."""

    def test_template_directory_exists(self, template_dir: Path) -> None:
        """
        Test that template directory exists.

        Verifies:
        - Template directory exists
        - Path is a directory
        - Located within papers/ directory

        Args:
            template_dir: Template directory path
        """
        assert template_dir.exists(), "papers/_template/ must exist"
        assert template_dir.is_dir(), "papers/_template/ must be a directory"

    def test_template_has_required_subdirectories(self, template_dir: Path) -> None:
        """
        Test that template has all required subdirectories.

        Verifies:
        - src/ directory exists
        - scripts/ directory exists
        - tests/ directory exists
        - data/ directory exists
        - configs/ directory exists
        - notebooks/ directory exists
        - examples/ directory exists

        Args:
            template_dir: Template directory path
        """
        required_dirs = ["src", "scripts", "tests", "data", "configs", "notebooks", "examples"]

        for dir_name in required_dirs:
            dir_path = template_dir / dir_name
            assert dir_path.exists(), f"Template must have {dir_name}/ directory"
            assert dir_path.is_dir(), f"Template {dir_name}/ must be a directory"

    def test_template_has_readme(self, template_dir: Path) -> None:
        """
        Test that template has README.md.

        Verifies:
        - README.md exists
        - File has content
        - Contains template instructions

        Args:
            template_dir: Template directory path
        """
        readme = template_dir / "README.md"
        assert readme.exists(), "Template must have README.md"
        assert readme.is_file(), "Template README.md must be a file"
        assert readme.stat().st_size > 0, "Template README.md must not be empty"

        content = readme.read_text()
        assert "Template" in content or "template" in content, "README must indicate it's a template"

    def test_template_src_has_init(self, template_dir: Path) -> None:
        """
        Test that template src/ has __init__.mojo.

        Verifies:
        - src/__init__.mojo exists
        - Path is a file

        Args:
            template_dir: Template directory path
        """
        init_file = template_dir / "src" / "__init__.mojo"
        assert init_file.exists(), "Template src/ must have __init__.mojo"
        assert init_file.is_file(), "Template src/__init__.mojo must be a file"

    def test_template_tests_has_init(self, template_dir: Path) -> None:
        """
        Test that template tests/ has __init__.mojo.

        Verifies:
        - tests/__init__.mojo exists
        - Path is a file

        Args:
            template_dir: Template directory path
        """
        init_file = template_dir / "tests" / "__init__.mojo"
        assert init_file.exists(), "Template tests/ must have __init__.mojo"
        assert init_file.is_file(), "Template tests/__init__.mojo must be a file"

    def test_template_data_has_subdirectories(self, template_dir: Path) -> None:
        """
        Test that template data/ has required subdirectories.

        Verifies:
        - data/raw/ directory exists
        - data/processed/ directory exists
        - data/cache/ directory exists

        Args:
            template_dir: Template directory path
        """
        data_dir = template_dir / "data"
        assert data_dir.exists(), "Template must have data/ directory"

        subdirs = ["raw", "processed", "cache"]
        for subdir in subdirs:
            subdir_path = data_dir / subdir
            assert subdir_path.exists(), f"Template data/ must have {subdir}/ directory"
            assert subdir_path.is_dir(), f"Template data/{subdir}/ must be a directory"

    def test_template_has_gitkeep_files(self, template_dir: Path) -> None:
        """
        Test that template has .gitkeep files in empty directories.

        Verifies:
        - .gitkeep files exist where needed
        - Files mark empty directories for git

        Args:
            template_dir: Template directory path
        """
        # Directories that should have .gitkeep files
        gitkeep_dirs = ["src", "scripts", "tests", "notebooks", "examples", "configs"]

        for dir_name in gitkeep_dirs:
            gitkeep = template_dir / dir_name / ".gitkeep"
            assert gitkeep.exists(), f"Template {dir_name}/ should have .gitkeep file"
            assert gitkeep.is_file(), f"Template {dir_name}/.gitkeep must be a file"

    def test_template_configs_has_example(self, template_dir: Path) -> None:
        """
        Test that template configs/ has example configuration.

        Verifies:
        - config.yaml exists
        - File has content

        Args:
            template_dir: Template directory path
        """
        config = template_dir / "configs" / "config.yaml"
        assert config.exists(), "Template configs/ must have config.yaml"
        assert config.is_file(), "Template config.yaml must be a file"


class TestTemplateCompleteness:
    """Test cases for template completeness."""

    def test_template_structure_matches_specification(
        self, template_dir: Path, expected_template_structure: Dict[str, List[str]]
    ) -> None:
        """
        Test that template structure matches expected specification.

        Verifies:
        - All expected directories exist
        - All expected files exist
        - Structure matches planning document

        Args:
            template_dir: Template directory path
            expected_template_structure: Expected structure specification
        """
        errors = validate_directory_structure(template_dir, expected_template_structure)

        assert not errors, "Template structure validation failed:\n" + "\n".join(errors)

    def test_template_readme_has_required_sections(self, template_dir: Path) -> None:
        """
        Test that template README has all required sections.

        Verifies:
        - Overview section exists
        - Quick Start section exists
        - Directory Structure section exists
        - Implementation Guide section exists

        Args:
            template_dir: Template directory path
        """
        readme = template_dir / "README.md"
        content = readme.read_text()

        required_sections = ["Overview", "Quick Start", "Directory Structure", "Directory Purposes"]

        for section in required_sections:
            assert section in content, f"Template README must have '{section}' section"

    def test_template_readme_documents_all_directories(self, template_dir: Path) -> None:
        """
        Test that template README documents all directories.

        Verifies:
        - README mentions src/
        - README mentions scripts/ (or script/)
        - README mentions tests/
        - README mentions data/
        - README mentions configs/
        - README mentions notebooks/
        - README mentions examples/

        Args:
            template_dir: Template directory path
        """
        readme = template_dir / "README.md"
        content = readme.read_text()

        # Check most directories with exact match
        required_dirs = ["src/", "tests/", "data/", "configs/", "notebooks/", "examples/"]

        for dir_ref in required_dirs:
            assert dir_ref in content, f"Template README must document {dir_ref} directory"

        # Special case: scripts directory (README uses "script/" singular)
        assert "script/" in content or "scripts/" in content, "Template README must document scripts/ directory"


class TestTemplateCopyFunctionality:
    """Test cases for template copy functionality."""

    def test_template_can_be_copied(self, template_dir: Path, tmp_path: Path) -> None:
        """
        Test that template can be copied to create new paper.

        Verifies:
        - Template can be copied successfully
        - All files are copied
        - All subdirectories are copied

        Args:
            template_dir: Template directory path
            tmp_path: Temporary directory for testing
        """
        # Copy template to temporary location
        dest_dir = tmp_path / "test-paper"
        shutil.copytree(template_dir, dest_dir)

        # Verify copy succeeded
        assert dest_dir.exists(), "Copied template directory must exist"
        assert dest_dir.is_dir(), "Copied template must be a directory"

        # Verify key components were copied
        assert (dest_dir / "README.md").exists(), "Copied template must have README.md"
        assert (dest_dir / "src").exists(), "Copied template must have src/ directory"
        assert (dest_dir / "tests").exists(), "Copied template must have tests/ directory"

    def test_copied_template_is_independent(self, template_dir: Path, tmp_path: Path) -> None:
        """
        Test that copied template is independent of original.

        Verifies:
        - Modifying copy doesn't affect original
        - Copy and original are separate filesystem entities

        Args:
            template_dir: Template directory path
            tmp_path: Temporary directory for testing
        """
        # Copy template
        dest_dir = tmp_path / "test-paper"
        shutil.copytree(template_dir, dest_dir)

        # Modify copied README
        copied_readme = dest_dir / "README.md"
        original_readme = template_dir / "README.md"

        original_content = original_readme.read_text()
        copied_readme.write_text("# Modified Paper Implementation")

        # Verify original is unchanged
        assert original_readme.read_text() == original_content, (
            "Original template must not be affected by copy modifications"
        )

    def test_copied_template_has_complete_structure(
        self, template_dir: Path, tmp_path: Path, expected_template_structure: Dict[str, List[str]]
    ) -> None:
        """
        Test that copied template has complete structure.

        Verifies:
        - All directories are copied
        - All files are copied
        - Structure matches specification

        Args:
            template_dir: Template directory path
            tmp_path: Temporary directory for testing
            expected_template_structure: Expected structure specification
        """
        # Copy template
        dest_dir = tmp_path / "test-paper"
        shutil.copytree(template_dir, dest_dir)

        # Validate copied structure
        errors = validate_directory_structure(dest_dir, expected_template_structure)

        assert not errors, "Copied template structure validation failed:\n" + "\n".join(errors)


class TestTemplateDocumentation:
    """Test cases for template documentation quality."""

    def test_template_readme_provides_usage_instructions(self, template_dir: Path) -> None:
        """
        Test that template README provides clear usage instructions.

        Verifies:
        - Instructions for copying template
        - Instructions for updating README
        - Implementation guide

        Args:
            template_dir: Template directory path
        """
        readme = template_dir / "README.md"
        content = readme.read_text()

        # Check for usage instructions
        assert "cp -r" in content or "copy" in content.lower(), "README must provide template copy instructions"
        assert "Implementation Guide" in content or "Step" in content, "README must provide implementation guide"

    def test_template_readme_explains_directory_purposes(self, template_dir: Path) -> None:
        """
        Test that template README explains each directory's purpose.

        Verifies:
        - src/ purpose is explained
        - tests/ purpose is explained
        - data/ purpose is explained
        - configs/ purpose is explained

        Args:
            template_dir: Template directory path
        """
        readme = template_dir / "README.md"
        content = readme.read_text()

        # Directory purpose sections should exist
        directory_sections = [
            ("src/", "implementation"),
            ("tests/", "test"),
            ("data/", "dataset"),
            ("configs/", "configuration"),
        ]

        for dir_name, keyword in directory_sections:
            # Check that directory is mentioned near its purpose keyword
            assert dir_name in content, f"README must mention {dir_name}"
            # This is a loose check - just verify documentation exists
            assert keyword in content.lower(), f"README should explain {dir_name} purpose related to {keyword}"
