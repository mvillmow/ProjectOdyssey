#!/usr/bin/env python3
"""
Unit tests for scripts/validation.py

Tests shared validation utilities.
"""

import pytest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.validation import (
    find_markdown_files,
    validate_file_exists,
    validate_directory_exists,
    check_required_sections,
    extract_markdown_links,
    count_markdown_issues,
)


class TestFindMarkdownFiles:
    """Test find_markdown_files() function"""

    def test_find_markdown_files_in_temp_dir(self):
        """Test finding markdown files in a temporary directory"""
        with TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create some markdown files
            (tmppath / "test1.md").write_text("# Test 1")
            (tmppath / "test2.md").write_text("# Test 2")
            (tmppath / "other.txt").write_text("Not markdown")

            files = find_markdown_files(tmppath)
            assert len(files) == 2
            assert all(f.suffix == ".md" for f in files)

    def test_find_markdown_files_excludes_dirs(self):
        """Test that excluded directories are skipped"""
        with TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create markdown in normal dir
            (tmppath / "good.md").write_text("# Good")

            # Create markdown in excluded dir
            node_modules = tmppath / "node_modules"
            node_modules.mkdir()
            (node_modules / "bad.md").write_text("# Bad")

            files = find_markdown_files(tmppath)
            assert len(files) == 1
            assert files[0].name == "good.md"


class TestValidateFileExists:
    """Test validate_file_exists() function"""

    def test_validate_existing_file(self):
        """Test validation of existing file"""
        # Use this test file itself
        assert validate_file_exists(Path(__file__))

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file"""
        assert not validate_file_exists(Path("/nonexistent/file.txt"))

    def test_validate_directory_as_file(self):
        """Test that directories are not considered files"""
        # Use parent directory
        assert not validate_file_exists(Path(__file__).parent)


class TestValidateDirectoryExists:
    """Test validate_directory_exists() function"""

    def test_validate_existing_directory(self):
        """Test validation of existing directory"""
        assert validate_directory_exists(Path(__file__).parent)

    def test_validate_nonexistent_directory(self):
        """Test validation of non-existent directory"""
        assert not validate_directory_exists(Path("/nonexistent/dir"))

    def test_validate_file_as_directory(self):
        """Test that files are not considered directories"""
        assert not validate_directory_exists(Path(__file__))


class TestCheckRequiredSections:
    """Test check_required_sections() function"""

    def test_check_all_sections_present(self):
        """Test when all required sections are present"""
        content = """
# Main Title

## Section 1

Content here.

## Section 2

More content.
"""
        required = ["Section 1", "Section 2"]
        all_found, missing = check_required_sections(content, required)
        assert all_found
        assert len(missing) == 0

    def test_check_missing_sections(self):
        """Test when some sections are missing"""
        content = """
# Main Title

## Section 1

Content here.
"""
        required = ["Section 1", "Section 2", "Section 3"]
        all_found, missing = check_required_sections(content, required)
        assert not all_found
        assert "Section 2" in missing
        assert "Section 3" in missing


class TestExtractMarkdownLinks:
    """Test extract_markdown_links() function"""

    def test_extract_simple_link(self):
        """Test extracting a simple markdown link"""
        content = "[Example](https://example.com)"
        links = extract_markdown_links(content)
        assert len(links) == 1
        assert links[0][0] == "https://example.com"
        assert links[0][1] == 1  # line number

    def test_extract_multiple_links(self):
        """Test extracting multiple links"""
        content = """
Line 1 has [Link 1](link1.md)
Line 2 has no links
Line 3 has [Link 2](link2.md) and [Link 3](link3.md)
"""
        links = extract_markdown_links(content)
        assert len(links) == 3


class TestCountMarkdownIssues:
    """Test count_markdown_issues() function"""

    def test_count_multiple_blank_lines(self):
        """Test counting multiple consecutive blank lines"""
        content = "Line 1\n\n\n\nLine 2"  # 3 blank lines
        issues = count_markdown_issues(content)
        assert issues["multiple_blank_lines"] > 0

    def test_count_missing_language_tags(self):
        """Test counting code blocks without language tags"""
        content = "```\ncode here\n```"
        issues = count_markdown_issues(content)
        assert issues["missing_language_tags"] > 0

    def test_count_long_lines(self):
        """Test counting lines over 120 characters"""
        content = "a" * 150  # Very long line
        issues = count_markdown_issues(content)
        assert issues["long_lines"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
