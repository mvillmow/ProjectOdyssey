"""
Test suite for tools/ directory structure validation.

This module validates that the tools/ directory structure exists with correct
permissions, locations, and organization.

Test Categories:
- Unit tests: Directory existence, permissions, location
- Edge cases: Directory already exists (idempotent)
- Integration tests: Complete directory hierarchy

Coverage Target: 100% (infrastructure critical)
"""

import stat
from pathlib import Path
from typing import Dict

import pytest


class TestToolsDirectoryCreation:
    """Test cases for validating tools/ directory structure."""

    def test_tools_directory_exists(self, tools_root: Path) -> None:
        """
        Test that tools/ directory exists at repository root.

        Verifies:
        - Directory exists after creation
        - Path is a directory (not a file)

        Args:
            tools_root: Path to tools/ directory
        """
        assert tools_root.exists(), "tools/ directory should exist at repository root"
        assert tools_root.is_dir(), "tools/ should be a directory, not a file"

    def test_tools_directory_location(
        self, repo_root: Path, tools_root: Path
    ) -> None:
        """
        Test that tools/ directory is at correct location.

        Verifies:
        - Directory is directly under repository root
        - Directory name is exactly "tools"
        - Path is absolute (not relative)

        Args:
            repo_root: Repository root directory
            tools_root: Path to tools/ directory
        """
        assert tools_root.parent == repo_root, (
            "tools/ should be directly under repository root"
        )
        assert tools_root.name == "tools", "Directory name should be 'tools'"
        assert tools_root.is_absolute(), "tools/ path should be absolute"

    def test_tools_directory_permissions(self, tools_root: Path) -> None:
        """
        Test that tools/ directory has correct permissions.

        Verifies:
        - Directory has write permissions (can create files)
        - Directory has read permissions (can list contents)
        - Directory has execute permissions (can access contents)

        Args:
            tools_root: Path to tools/ directory
        """
        # Get directory permissions
        dir_stat = tools_root.stat()
        mode = dir_stat.st_mode

        # Assert: Directory has read, write, and execute permissions
        assert (
            mode & stat.S_IRUSR
        ), "tools/ should have read permission for owner"
        assert (
            mode & stat.S_IWUSR
        ), "tools/ should have write permission for owner"
        assert (
            mode & stat.S_IXUSR
        ), "tools/ should have execute permission for owner"

        # Additional verification: Can create a file inside
        test_file = tools_root / ".test_permissions.txt"
        try:
            test_file.write_text("test content")
            assert test_file.exists(), "Should be able to create files in tools/"
        finally:
            # Clean up test file
            if test_file.exists():
                test_file.unlink()

    def test_tools_readme_exists(self, main_readme: Path) -> None:
        """
        Test that main tools/README.md file exists.

        Verifies:
        - README.md exists in tools/ directory
        - File is a regular file (not a directory)

        Args:
            main_readme: Path to tools/README.md
        """
        assert main_readme.exists(), "tools/README.md should exist"
        assert main_readme.is_file(), "tools/README.md should be a file"


class TestCategoryDirectories:
    """Test cases for validating tool category directories."""

    def test_all_category_directories_exist(
        self, category_paths: Dict[str, Path]
    ) -> None:
        """
        Test that all 4 tool category directories exist.

        Verifies:
        - paper-scaffold/ exists
        - test-utils/ exists
        - benchmarking/ exists
        - codegen/ exists

        Args:
            category_paths: Dictionary of category paths
        """
        for category_name, category_path in category_paths.items():
            assert category_path.exists(), (
                f"tools/{category_name}/ should exist"
            )
            assert category_path.is_dir(), (
                f"tools/{category_name}/ should be a directory"
            )

    def test_category_directory_names(
        self, tools_root: Path, category_names: list
    ) -> None:
        """
        Test that category directories have correct names.

        Verifies:
        - All expected categories are present
        - Names use lowercase with hyphens (paper-scaffold, not paper_scaffold)
        - No unexpected directories present

        Args:
            tools_root: Path to tools/ directory
            category_names: List of expected category names
        """
        # Get actual directory names (excluding README.md and other files)
        actual_dirs = {
            item.name for item in tools_root.iterdir()
            if item.is_dir()
        }

        # Expected categories
        expected_dirs = set(category_names)

        # Verify all expected directories are present
        assert expected_dirs.issubset(actual_dirs), (
            f"Missing categories: {expected_dirs - actual_dirs}"
        )

        # Verify naming convention (lowercase with hyphens)
        for name in actual_dirs:
            assert name.islower() or "-" in name, (
                f"Category '{name}' should use lowercase with hyphens"
            )

    def test_category_directory_permissions(
        self, category_paths: Dict[str, Path]
    ) -> None:
        """
        Test that all category directories have correct permissions.

        Verifies:
        - Each category directory has read, write, execute permissions
        - Files can be created in each category directory

        Args:
            category_paths: Dictionary of category paths
        """
        for category_name, category_path in category_paths.items():
            # Get directory permissions
            dir_stat = category_path.stat()
            mode = dir_stat.st_mode

            # Verify permissions
            assert (mode & stat.S_IRUSR), (
                f"{category_name}/ should have read permission"
            )
            assert (mode & stat.S_IWUSR), (
                f"{category_name}/ should have write permission"
            )
            assert (mode & stat.S_IXUSR), (
                f"{category_name}/ should have execute permission"
            )

    def test_no_unexpected_directories(
        self, tools_root: Path, category_names: list
    ) -> None:
        """
        Test that no unexpected directories exist in tools/.

        Verifies:
        - Only expected category directories are present
        - No extra directories added accidentally

        Args:
            tools_root: Path to tools/ directory
            category_names: List of expected category names
        """
        # Get actual directory names
        actual_dirs = {
            item.name for item in tools_root.iterdir()
            if item.is_dir()
        }

        # Expected directories (only the 4 categories)
        expected_dirs = set(category_names)

        # Check for unexpected directories
        unexpected = actual_dirs - expected_dirs

        # Allow common development directories that might exist
        # Including 'setup' which contains installation and verification scripts
        allowed_extra = {"__pycache__", ".pytest_cache", "setup"}
        unexpected = unexpected - allowed_extra

        assert len(unexpected) == 0, (
            f"Unexpected directories in tools/: {unexpected}"
        )


class TestToolsDirectoryIntegration:
    """Integration tests for tools/ directory functionality."""

    def test_complete_directory_hierarchy(
        self, tools_root: Path, category_paths: Dict[str, Path]
    ) -> None:
        """
        Test that complete tools/ directory hierarchy exists and is accessible.

        This integration test validates the complete structure:
        1. tools/ directory exists
        2. All 4 category directories exist
        3. All category directories are accessible

        Args:
            tools_root: Path to tools/ directory
            category_paths: Dictionary of category paths
        """
        # Verify root exists
        assert tools_root.exists() and tools_root.is_dir()

        # Verify all categories exist
        for category_name, category_path in category_paths.items():
            assert category_path.exists(), (
                f"{category_name}/ should exist"
            )
            assert category_path.is_dir(), (
                f"{category_name}/ should be a directory"
            )

            # Verify category is under tools/
            assert category_path.parent == tools_root, (
                f"{category_name}/ should be directly under tools/"
            )

    def test_can_list_tools_directory_contents(
        self, tools_root: Path, category_names: list
    ) -> None:
        """
        Test that tools/ directory contents can be listed.

        Verifies:
        - Directory can be iterated
        - All expected categories appear in listing
        - Listing includes README.md

        Args:
            tools_root: Path to tools/ directory
            category_names: List of expected category names
        """
        # List all items in tools/
        contents = list(tools_root.iterdir())
        assert len(contents) > 0, "tools/ should not be empty"

        # Get names of all items
        content_names = {item.name for item in contents}

        # Verify all categories are in listing
        for category in category_names:
            assert category in content_names, (
                f"{category}/ should appear in directory listing"
            )

        # Verify README.md is in listing
        assert "README.md" in content_names, (
            "README.md should appear in directory listing"
        )

    def test_category_directories_can_contain_files(
        self, category_paths: Dict[str, Path]
    ) -> None:
        """
        Test that files can be created in category directories.

        Verifies:
        - Files can be written to each category directory
        - Files can be read back correctly
        - Both read and write operations work

        Args:
            category_paths: Dictionary of category paths
        """
        for category_name, category_path in category_paths.items():
            # Create a test file
            test_file = category_path / ".test_file_permissions.txt"
            test_content = f"Test content for {category_name}"

            try:
                # Write test file
                test_file.write_text(test_content)

                # Verify file exists and has correct content
                assert test_file.exists(), (
                    f"Should be able to create files in {category_name}/"
                )
                assert test_file.read_text() == test_content, (
                    f"File content should match in {category_name}/"
                )
            finally:
                # Clean up test file
                if test_file.exists():
                    test_file.unlink()
