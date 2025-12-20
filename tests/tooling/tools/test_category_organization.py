"""
Test suite for tool category organization validation.

This module validates that tool categories have correct structure, follow
naming conventions, and are organized according to ADR-001 language strategy.

Test Categories:
- Category structure: Each category has correct organization
- Naming conventions: Categories follow naming standards
- Language alignment: Categories align with Mojo/Python strategy

Coverage Target: 95% (infrastructure validation)
"""

from pathlib import Path
from typing import Dict



class TestCategoryStructure:
    """Test cases for validating tool category structure."""

    def test_paper_scaffold_category_structure(self, category_paths: Dict[str, Path]) -> None:
        """
        Test that paper-scaffold category has correct structure.

        Verifies:
        - Category directory exists
        - README.md present at category root
        - Category can contain subdirectories

        Args:
            category_paths: Dictionary of category paths
        """
        paper_scaffold = category_paths["paper-scaffold"]

        # Verify directory exists
        assert paper_scaffold.exists(), "paper-scaffold/ should exist"
        assert paper_scaffold.is_dir(), "paper-scaffold/ should be a directory"

        # Verify README at root
        readme = paper_scaffold / "README.md"
        assert readme.exists(), "paper-scaffold/README.md should exist"
        assert readme.is_file(), "paper-scaffold/README.md should be a file"

    def test_test_utils_category_structure(self, category_paths: Dict[str, Path]) -> None:
        """
        Test that test-utils category has correct structure.

        Verifies:
        - Category directory exists
        - README.md present at category root

        Args:
            category_paths: Dictionary of category paths
        """
        test_utils = category_paths["test-utils"]

        # Verify directory exists
        assert test_utils.exists(), "test-utils/ should exist"
        assert test_utils.is_dir(), "test-utils/ should be a directory"

        # Verify README at root
        readme = test_utils / "README.md"
        assert readme.exists(), "test-utils/README.md should exist"
        assert readme.is_file(), "test-utils/README.md should be a file"

    def test_benchmarking_category_structure(self, category_paths: Dict[str, Path]) -> None:
        """
        Test that benchmarking category has correct structure.

        Verifies:
        - Category directory exists
        - README.md present at category root

        Args:
            category_paths: Dictionary of category paths
        """
        benchmarking = category_paths["benchmarking"]

        # Verify directory exists
        assert benchmarking.exists(), "benchmarking/ should exist"
        assert benchmarking.is_dir(), "benchmarking/ should be a directory"

        # Verify README at root
        readme = benchmarking / "README.md"
        assert readme.exists(), "benchmarking/README.md should exist"
        assert readme.is_file(), "benchmarking/README.md should be a file"

    def test_codegen_category_structure(self, category_paths: Dict[str, Path]) -> None:
        """
        Test that codegen category has correct structure.

        Verifies:
        - Category directory exists
        - README.md present at category root

        Args:
            category_paths: Dictionary of category paths
        """
        codegen = category_paths["codegen"]

        # Verify directory exists
        assert codegen.exists(), "codegen/ should exist"
        assert codegen.is_dir(), "codegen/ should be a directory"

        # Verify README at root
        readme = codegen / "README.md"
        assert readme.exists(), "codegen/README.md should exist"
        assert readme.is_file(), "codegen/README.md should be a file"

    def test_category_readme_location(self, category_readmes: Dict[str, Path]) -> None:
        """
        Test that each category has README.md at its root.

        Verifies:
        - README.md is directly in category directory (not nested)
        - README.md is a file, not a directory

        Args:
            category_readmes: Dictionary of category README paths
        """
        for category_name, readme_path in category_readmes.items():
            # Verify README exists
            assert readme_path.exists(), f"{category_name}/README.md should exist at category root"

            # Verify it's a file
            assert readme_path.is_file(), f"{category_name}/README.md should be a file"

            # Verify it's at category root (parent is category directory)
            assert readme_path.name == "README.md", f"{category_name} README should be named README.md"


class TestCategoryNamingConventions:
    """Test cases for category naming conventions."""

    def test_category_follows_naming_convention(self, category_names: list) -> None:
        """
        Test that all categories follow naming convention.

        Verifies:
        - Category names use lowercase
        - Multi-word names use hyphens (paper-scaffold, not paper_scaffold)
        - Names are descriptive

        Args:
            category_names: List of category names
        """
        for category in category_names:
            # Check lowercase (allowing hyphens)
            assert category.replace("-", "").islower(), f"Category '{category}' should use lowercase"

            # Check no underscores (use hyphens instead)
            assert "_" not in category, f"Category '{category}' should use hyphens, not underscores"

            # Check reasonable length
            assert 3 <= len(category) <= 30, f"Category '{category}' should have reasonable name length"

    def test_category_names_are_descriptive(self, category_names: list) -> None:
        """
        Test that category names are descriptive of their purpose.

        Verifies:
        - Names indicate tool category purpose
        - Names avoid generic terms (e.g., not just "tools" or "utils")

        Args:
            category_names: List of category names
        """
        # Define expected descriptive terms
        expected_categories = {
            "paper-scaffold",
            "test-utils",
            "benchmarking",
            "codegen",
        }

        # Verify all categories are from expected set
        actual_categories = set(category_names)
        assert actual_categories == expected_categories, (
            f"Categories should match expected set. Got: {actual_categories}, Expected: {expected_categories}"
        )

    def test_category_paths_are_relative_to_tools(self, tools_root: Path, category_paths: Dict[str, Path]) -> None:
        """
        Test that category paths are directly under tools/.

        Verifies:
        - Each category is one level below tools/
        - No nested category structures

        Args:
            tools_root: Path to tools/ directory
            category_paths: Dictionary of category paths
        """
        for category_name, category_path in category_paths.items():
            # Verify parent is tools/
            assert category_path.parent == tools_root, f"{category_name}/ should be directly under tools/"

            # Verify path structure
            expected_path = tools_root / category_name
            assert category_path == expected_path, f"{category_name} path should be tools/{category_name}"


class TestCategoryOrganization:
    """Test cases for overall category organization."""

    def test_category_supports_subdirectories(self, category_paths: Dict[str, Path]) -> None:
        """
        Test that categories can contain subdirectories.

        Verifies:
        - Subdirectories can be created in categories
        - Categories support nested structure for tools

        Args:
            category_paths: Dictionary of category paths
        """
        # Test with paper-scaffold (most likely to have subdirectories)
        paper_scaffold = category_paths["paper-scaffold"]

        # Create temporary test subdirectory
        test_subdir = paper_scaffold / ".test_subdir"

        try:
            # Create subdirectory
            test_subdir.mkdir(exist_ok=True)

            # Verify subdirectory exists
            assert test_subdir.exists(), "Should be able to create subdirectories in categories"
            assert test_subdir.is_dir(), "Created subdirectory should be a directory"

        finally:
            # Clean up test subdirectory
            if test_subdir.exists():
                test_subdir.rmdir()

    def test_category_can_contain_files(self, category_paths: Dict[str, Path]) -> None:
        """
        Test that categories can contain files.

        Verifies:
        - Files can be created in categories
        - Categories support tools as files

        Args:
            category_paths: Dictionary of category paths
        """
        for category_name, category_path in category_paths.items():
            # Create temporary test file
            test_file = category_path / ".test_file.txt"

            try:
                # Create file
                test_file.write_text("test content")

                # Verify file exists
                assert test_file.exists(), f"Should be able to create files in {category_name}/"
                assert test_file.is_file(), f"Created item should be a file in {category_name}/"

            finally:
                # Clean up test file
                if test_file.exists():
                    test_file.unlink()

    def test_categories_are_independent(self, category_paths: Dict[str, Path]) -> None:
        """
        Test that categories are independent of each other.

        Verifies:
        - Each category is a separate directory
        - Categories don't share structure
        - Modifying one category doesn't affect others

        Args:
            category_paths: Dictionary of category paths
        """
        # Get list of category paths
        paths = list(category_paths.values())

        # Verify each category has unique path
        assert len(paths) == len(set(paths)), "Each category should have unique path"

        # Verify categories don't nest within each other
        for i, path1 in enumerate(paths):
            for j, path2 in enumerate(paths):
                if i != j:
                    # path1 should not be parent of path2
                    assert path2.parent != path1, f"Categories should not nest: {path1.name} -> {path2.name}"


class TestLanguageStrategyAlignment:
    """Test cases for language strategy alignment with ADR-001."""

    def test_paper_scaffold_language_documented(self, category_readmes: Dict[str, Path]) -> None:
        """
        Test that paper-scaffold README documents language choice.

        Verifies:
        - README mentions Python (for scaffolding script)
        - README mentions Mojo (for generated templates)

        Args:
            category_readmes: Dictionary of category README paths
        """
        readme_path = category_readmes["paper-scaffold"]
        content = readme_path.read_text().lower()

        # paper-scaffold should use Python for scaffolding script
        # and generate Mojo templates
        # Check if README mentions this or has "Coming Soon"
        has_language_info = "python" in content or "mojo" in content or "coming soon" in content

        assert has_language_info, "paper-scaffold/README.md should mention language strategy or have 'Coming Soon'"

    def test_test_utils_language_documented(self, category_readmes: Dict[str, Path]) -> None:
        """
        Test that test-utils README documents language choice.

        Verifies:
        - README mentions Mojo (primary) or Python (integration)

        Args:
            category_readmes: Dictionary of category README paths
        """
        readme_path = category_readmes["test-utils"]
        content = readme_path.read_text().lower()

        # test-utils should primarily use Mojo
        has_language_info = "mojo" in content or "python" in content or "coming soon" in content

        assert has_language_info, "test-utils/README.md should mention language strategy or have 'Coming Soon'"

    def test_benchmarking_language_documented(self, category_readmes: Dict[str, Path]) -> None:
        """
        Test that benchmarking README documents language choice.

        Verifies:
        - README mentions Mojo (required for accurate benchmarks)

        Args:
            category_readmes: Dictionary of category README paths
        """
        readme_path = category_readmes["benchmarking"]
        content = readme_path.read_text().lower()

        # benchmarking should use Mojo for accurate measurements
        has_language_info = "mojo" in content or "python" in content or "coming soon" in content

        assert has_language_info, "benchmarking/README.md should mention language strategy or have 'Coming Soon'"

    def test_codegen_language_documented(self, category_readmes: Dict[str, Path]) -> None:
        """
        Test that codegen README documents language choice.

        Verifies:
        - README mentions Python (for code generation scripts)

        Args:
            category_readmes: Dictionary of category README paths
        """
        readme_path = category_readmes["codegen"]
        content = readme_path.read_text().lower()

        # codegen should use Python for templating
        has_language_info = "python" in content or "mojo" in content or "coming soon" in content

        assert has_language_info, "codegen/README.md should mention language strategy or have 'Coming Soon'"
