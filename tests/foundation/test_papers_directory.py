"""
Test suite for papers/ directory creation.

This module contains comprehensive tests for creating and verifying the papers/
directory at the repository root, following TDD/FIRST principles.

Test Categories:
- Unit tests: Directory creation, permissions, location
- Edge cases: Already exists, permission denied, parent missing
- Integration tests: Subdirectory creation

Coverage Target: >95%
"""

import stat
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_repo_root(tmp_path: Path) -> Path:
    """
    Provide a mock repository root directory for testing.

    Args:
        tmp_path: pytest built-in fixture providing temporary directory

    Returns:
        Path to temporary directory acting as repository root
    """
    return tmp_path


@pytest.fixture
def papers_dir(mock_repo_root: Path) -> Path:
    """
    Provide the expected papers directory path.

    Args:
        mock_repo_root: Temporary repository root directory

    Returns:
        Path to papers directory within repository root
    """
    return mock_repo_root / "papers"


class TestPapersDirectoryCreation:
    """Test cases for creating the papers/ directory."""

    def test_create_papers_directory_success(self, mock_repo_root: Path, papers_dir: Path) -> None:
        """
        Test successful creation of papers/ directory.

        Verifies:
        - Directory is created successfully
        - Path exists after creation
        - Created path is a directory (not a file)

        Args:
            mock_repo_root: Temporary repository root
            papers_dir: Expected papers directory path
        """
        # Act: Create the papers directory
        papers_dir.mkdir(parents=True, exist_ok=True)

        # Assert: Directory exists and is a directory
        assert papers_dir.exists(), "papers/ directory should exist after creation"
        assert papers_dir.is_dir(), "papers/ should be a directory, not a file"

    def test_papers_directory_permissions(self, mock_repo_root: Path, papers_dir: Path) -> None:
        """
        Test that papers/ directory has correct permissions.

        Verifies:
        - Directory has write permissions (can create files)
        - Directory has read permissions (can list contents)
        - Directory has execute permissions (can access contents)

        Args:
            mock_repo_root: Temporary repository root
            papers_dir: Expected papers directory path
        """
        # Arrange: Create the directory
        papers_dir.mkdir(parents=True, exist_ok=True)

        # Act: Get directory permissions
        dir_stat = papers_dir.stat()
        mode = dir_stat.st_mode

        # Assert: Directory has read, write, and execute permissions
        assert mode & stat.S_IRUSR, "papers/ should have read permission for owner"
        assert mode & stat.S_IWUSR, "papers/ should have write permission for owner"
        assert mode & stat.S_IXUSR, "papers/ should have execute permission for owner"

        # Additional verification: Can create a file inside
        test_file = papers_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists(), "Should be able to create files in papers/"
        test_file.unlink()  # Clean up

    def test_papers_directory_location(self, mock_repo_root: Path, papers_dir: Path) -> None:
        """
        Test that papers/ directory is created at correct location.

        Verifies:
        - Directory is directly under repository root
        - Directory name is exactly "papers"
        - Path is absolute (not relative)

        Args:
            mock_repo_root: Temporary repository root
            papers_dir: Expected papers directory path
        """
        # Arrange: Create the directory
        papers_dir.mkdir(parents=True, exist_ok=True)

        # Assert: Directory location is correct
        assert papers_dir.parent == mock_repo_root, "papers/ should be directly under repository root"
        assert papers_dir.name == "papers", "Directory name should be 'papers'"
        assert papers_dir.is_absolute(), "papers/ path should be absolute"


class TestPapersDirectoryEdgeCases:
    """Test edge cases and error conditions for papers/ directory creation."""

    def test_create_papers_directory_already_exists(self, mock_repo_root: Path, papers_dir: Path) -> None:
        """
        Test idempotent behavior when directory already exists.

        Verifies:
        - Creating directory twice doesn't raise error
        - Directory still exists after second creation
        - exist_ok=True parameter works correctly

        Args:
            mock_repo_root: Temporary repository root
            papers_dir: Expected papers directory path
        """
        # Arrange: Create directory first time
        papers_dir.mkdir(parents=True, exist_ok=True)
        assert papers_dir.exists(), "Directory should exist after first creation"

        # Act: Create directory second time (should not raise error)
        papers_dir.mkdir(parents=True, exist_ok=True)

        # Assert: Directory still exists
        assert papers_dir.exists(), "Directory should still exist after second creation"
        assert papers_dir.is_dir(), "papers/ should still be a directory"

    def test_create_papers_directory_parent_missing(self, tmp_path: Path) -> None:
        """
        Test directory creation when parent directories need to be created.

        Verifies:
        - parents=True creates intermediate directories
        - Nested path creation works correctly
        - All parent directories are created

        Args:
            tmp_path: Temporary directory for testing
        """
        # Arrange: Create nested path with non-existent parents
        nested_papers = tmp_path / "a" / "b" / "c" / "papers"

        # Act: Create directory with parents=True
        nested_papers.mkdir(parents=True, exist_ok=True)

        # Assert: All parent directories and target directory exist
        assert (tmp_path / "a").exists(), "First parent should be created"
        assert (tmp_path / "a" / "b").exists(), "Second parent should be created"
        assert (tmp_path / "a" / "b" / "c").exists(), "Third parent should be created"
        assert nested_papers.exists(), "Target directory should be created"
        assert nested_papers.is_dir(), "Target should be a directory"

    def test_create_papers_directory_permission_denied(self, mock_repo_root: Path, papers_dir: Path) -> None:
        """
        Test handling of permission denied errors.

        Verifies:
        - PermissionError is raised when permissions are denied
        - Error message is clear and informative
        - Original error is propagated correctly

        Args:
            mock_repo_root: Temporary repository root
            papers_dir: Expected papers directory path
        """
        # Arrange: Mock mkdir to raise PermissionError
        with patch.object(Path, "mkdir", side_effect=PermissionError("Permission denied: cannot create directory")):
            # Act & Assert: Attempting to create directory should raise PermissionError
            with pytest.raises(PermissionError) as exc_info:
                papers_dir.mkdir(parents=True, exist_ok=True)

            # Verify error message
            assert "Permission denied" in str(exc_info.value), "Error message should indicate permission was denied"

    def test_create_papers_directory_without_exist_ok(self, mock_repo_root: Path, papers_dir: Path) -> None:
        """
        Test that FileExistsError is raised when exist_ok=False.

        Verifies:
        - FileExistsError is raised when directory exists and exist_ok=False
        - Error indicates the directory already exists
        - Behavior matches expected Python Path.mkdir() semantics

        Args:
            mock_repo_root: Temporary repository root
            papers_dir: Expected papers directory path
        """
        # Arrange: Create directory first time
        papers_dir.mkdir(parents=True, exist_ok=True)

        # Act & Assert: Creating again with exist_ok=False should raise error
        with pytest.raises(FileExistsError):
            papers_dir.mkdir(parents=True, exist_ok=False)

        # Verify error is about file existing
        assert papers_dir.exists(), "Directory should still exist after error"


class TestPapersDirectoryIntegration:
    """Integration tests for papers/ directory functionality."""

    def test_can_create_subdirectory_in_papers(self, mock_repo_root: Path, papers_dir: Path) -> None:
        """
        Test creating subdirectories within papers/.

        Verifies:
        - Subdirectories can be created successfully
        - Multiple subdirectories can coexist
        - Subdirectories have correct parent relationship

        Args:
            mock_repo_root: Temporary repository root
            papers_dir: Expected papers directory path
        """
        # Arrange: Create papers directory
        papers_dir.mkdir(parents=True, exist_ok=True)

        # Act: Create subdirectories for paper implementations
        lenet5_dir = papers_dir / "lenet5"
        alexnet_dir = papers_dir / "alexnet"
        resnet_dir = papers_dir / "resnet"

        lenet5_dir.mkdir(parents=True, exist_ok=True)
        alexnet_dir.mkdir(parents=True, exist_ok=True)
        resnet_dir.mkdir(parents=True, exist_ok=True)

        # Assert: All subdirectories exist
        assert lenet5_dir.exists(), "lenet5 subdirectory should exist"
        assert alexnet_dir.exists(), "alexnet subdirectory should exist"
        assert resnet_dir.exists(), "resnet subdirectory should exist"

        # Assert: All are directories
        assert lenet5_dir.is_dir(), "lenet5 should be a directory"
        assert alexnet_dir.is_dir(), "alexnet should be a directory"
        assert resnet_dir.is_dir(), "resnet should be a directory"

        # Assert: All have correct parent
        assert lenet5_dir.parent == papers_dir, "lenet5 parent should be papers/"
        assert alexnet_dir.parent == papers_dir, "alexnet parent should be papers/"
        assert resnet_dir.parent == papers_dir, "resnet parent should be papers/"

    def test_papers_directory_can_contain_files(self, mock_repo_root: Path, papers_dir: Path) -> None:
        """
        Test that files can be created within papers/ directory.

        Verifies:
        - Files can be written to papers/ directory
        - File contents can be read back correctly
        - Both read and write operations work

        Args:
            mock_repo_root: Temporary repository root
            papers_dir: Expected papers directory path
        """
        # Arrange: Create papers directory
        papers_dir.mkdir(parents=True, exist_ok=True)

        # Act: Create a file in papers directory
        readme_file = papers_dir / "README.md"
        test_content = "# Papers\n\nThis directory contains paper implementations."
        readme_file.write_text(test_content)

        # Assert: File exists and has correct content
        assert readme_file.exists(), "README.md should exist in papers/"
        assert readme_file.is_file(), "README.md should be a file"
        assert readme_file.read_text() == test_content, "File content should match what was written"

    def test_papers_directory_listing(self, mock_repo_root: Path, papers_dir: Path) -> None:
        """
        Test listing contents of papers/ directory.

        Verifies:
        - Empty directory can be listed
        - Directory with contents can be listed
        - Listed items match what was created

        Args:
            mock_repo_root: Temporary repository root
            papers_dir: Expected papers directory path
        """
        # Arrange: Create papers directory
        papers_dir.mkdir(parents=True, exist_ok=True)

        # Assert: Empty directory listing works
        contents = list(papers_dir.iterdir())
        assert len(contents) == 0, "New papers/ directory should be empty"

        # Act: Add some subdirectories and files
        (papers_dir / "lenet5").mkdir()
        (papers_dir / "alexnet").mkdir()
        (papers_dir / "README.md").write_text("# Papers")

        # Assert: Directory listing shows all items
        contents = list(papers_dir.iterdir())
        assert len(contents) == 3, "papers/ should contain 3 items"

        content_names = {item.name for item in contents}
        assert content_names == {"lenet5", "alexnet", "README.md"}, "Directory should contain expected items"


class TestPapersDirectoryRealWorld:
    """Real-world scenario tests for papers/ directory."""

    def test_complete_workflow_papers_directory(self, mock_repo_root: Path, papers_dir: Path) -> None:
        """
        Test complete workflow of creating and using papers/ directory.

        This integration test simulates the real-world usage pattern:
        1. Create papers/ directory
        2. Create subdirectory for a paper (lenet5)
        3. Create source files in the paper directory
        4. Verify all operations succeed

        Args:
            mock_repo_root: Temporary repository root
            papers_dir: Expected papers directory path
        """
        # Step 1: Create papers directory
        papers_dir.mkdir(parents=True, exist_ok=True)
        assert papers_dir.exists() and papers_dir.is_dir()

        # Step 2: Create subdirectory for lenet5 paper
        lenet5_dir = papers_dir / "lenet5"
        lenet5_dir.mkdir(parents=True, exist_ok=True)
        assert lenet5_dir.exists() and lenet5_dir.is_dir()

        # Step 3: Create source files
        src_dir = lenet5_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)

        model_file = src_dir / "model.py"
        model_file.write_text("# LeNet-5 Model Implementation")

        train_file = src_dir / "train.py"
        train_file.write_text("# Training Script")

        # Step 4: Create documentation
        readme = lenet5_dir / "README.md"
        readme.write_text("# LeNet-5 Implementation")

        # Verify: Complete structure exists
        assert (papers_dir / "lenet5" / "src" / "model.py").exists()
        assert (papers_dir / "lenet5" / "src" / "train.py").exists()
        assert (papers_dir / "lenet5" / "README.md").exists()

        # Verify: Can list all structure
        all_files = list(lenet5_dir.rglob("*"))
        file_count = len([f for f in all_files if f.is_file()])
        assert file_count == 3, "Should have 3 files in lenet5 structure"
