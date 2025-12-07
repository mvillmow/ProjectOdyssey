#!/usr/bin/env python3
"""
Unit tests for scripts/validation.py

Tests the shared validation framework used across multiple scripts.
"""

import sys
import unittest
from pathlib import Path
import tempfile
import shutil

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from validation import (
    find_markdown_files,
    validate_file_exists,
    validate_directory_exists,
    extract_markdown_links,
    validate_relative_link,
    count_markdown_issues
)


class TestFindMarkdownFiles(unittest.TestCase):
    """Test suite for find_markdown_files function."""

    def setUp(self):
        """Create temporary directory for testing."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_find_empty_directory(self):
        """Test finding markdown files in empty directory."""
        files = find_markdown_files(self.test_dir)
        self.assertEqual(len(files), 0)

    def test_find_single_markdown_file(self):
        """Test finding a single markdown file."""
        md_file = self.test_dir / "test.md"
        md_file.write_text("# Test")

        files = find_markdown_files(self.test_dir)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].name, "test.md")

    def test_find_multiple_markdown_files(self):
        """Test finding multiple markdown files."""
        (self.test_dir / "test1.md").write_text("# Test 1")
        (self.test_dir / "test2.md").write_text("# Test 2")
        (self.test_dir / "README.md").write_text("# README")

        files = find_markdown_files(self.test_dir)
        self.assertEqual(len(files), 3)

    def test_find_nested_markdown_files(self):
        """Test finding markdown files in subdirectories."""
        subdir = self.test_dir / "subdir"
        subdir.mkdir()

        (self.test_dir / "root.md").write_text("# Root")
        (subdir / "nested.md").write_text("# Nested")

        files = find_markdown_files(self.test_dir)
        self.assertEqual(len(files), 2)

    def test_exclude_directories(self):
        """Test that excluded directories are skipped."""
        # Create node_modules and .git directories
        (self.test_dir / "node_modules").mkdir()
        (self.test_dir / ".git").mkdir()
        (self.test_dir / "regular").mkdir()

        (self.test_dir / "node_modules" / "excluded.md").write_text("# Excluded")
        (self.test_dir / ".git" / "excluded.md").write_text("# Excluded")
        (self.test_dir / "regular" / "included.md").write_text("# Included")

        files = find_markdown_files(self.test_dir)
        # Should only find the file in 'regular' directory
        self.assertEqual(len(files), 1)
        self.assertTrue(any("included.md" in str(f) for f in files))


class TestFileValidation(unittest.TestCase):
    """Test suite for file and directory validation functions."""

    def setUp(self):
        """Create temporary directory for testing."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_validate_existing_file(self):
        """Test validating an existing file."""
        test_file = self.test_dir / "test.txt"
        test_file.write_text("test content")

        result = validate_file_exists(test_file)
        self.assertTrue(result)

    def test_validate_missing_file(self):
        """Test validating a missing file."""
        test_file = self.test_dir / "missing.txt"

        result = validate_file_exists(test_file)
        self.assertFalse(result)

    def test_validate_existing_directory(self):
        """Test validating an existing directory."""
        test_dir = self.test_dir / "subdir"
        test_dir.mkdir()

        result = validate_directory_exists(test_dir)
        self.assertTrue(result)

    def test_validate_missing_directory(self):
        """Test validating a missing directory."""
        test_dir = self.test_dir / "missing"

        result = validate_directory_exists(test_dir)
        self.assertFalse(result)


class TestExtractMarkdownLinks(unittest.TestCase):
    """Test suite for markdown link extraction."""

    def test_extract_simple_link(self):
        """Test extracting a simple markdown link."""
        content = "[Link Text](https://example.com)"
        links = extract_markdown_links(content)

        self.assertEqual(len(links), 1)
        # Returns (link_target, line_number)
        self.assertEqual(links[0][0], "https://example.com")
        self.assertEqual(links[0][1], 1)

    def test_extract_multiple_links(self):
        """Test extracting multiple links."""
        content = """
        [Link 1](https://example.com)
        [Link 2](./relative/path.md)
        [Link 3](#anchor).
       """
        links = extract_markdown_links(content)

        self.assertEqual(len(links), 3)

    def test_extract_relative_file_link(self):
        """Test extracting relative file links."""
        content = "[Relative](./docs/README.md)"
        links = extract_markdown_links(content)

        self.assertEqual(len(links), 1)
        # Returns (link_target, line_number)
        self.assertEqual(links[0][0], "./docs/README.md")
        self.assertEqual(links[0][1], 1)

    def test_extract_anchor_link(self):
        """Test extracting anchor links."""
        content = "[Anchor](#section-title)"
        links = extract_markdown_links(content)

        self.assertEqual(len(links), 1)
        # Returns (link_target, line_number)
        self.assertEqual(links[0][0], "#section-title")
        self.assertEqual(links[0][1], 1)

    def test_extract_no_links(self):
        """Test content with no links."""
        content = "This is plain text with no links."
        links = extract_markdown_links(content)

        self.assertEqual(len(links), 0)


class TestValidateRelativeLink(unittest.TestCase):
    """Test suite for relative link validation."""

    def setUp(self):
        """Create temporary directory for testing."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.source_file = self.test_dir / "source.md"
        self.source_file.write_text("# Source")

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_validate_existing_relative_link(self):
        """Test validating a link to an existing file."""
        target_file = self.test_dir / "target.md"
        target_file.write_text("# Target")

        is_valid, error = validate_relative_link(
            "target.md",
            self.source_file,
            self.test_dir
        )

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_validate_missing_relative_link(self):
        """Test validating a link to a missing file."""
        is_valid, error = validate_relative_link(
            "missing.md",
            self.source_file,
            self.test_dir
        )

        self.assertFalse(is_valid)
        self.assertIn("not found", error.lower())

    def test_validate_anchor_only_link(self):
        """Test validating anchor-only links (should be skipped)."""
        is_valid, error = validate_relative_link(
            "#section",
            self.source_file,
            self.test_dir
        )

        # Anchor-only links should be valid (skipped)
        self.assertTrue(is_valid)


class TestCountMarkdownIssues(unittest.TestCase):
    """Test suite for counting markdown issues."""

    def test_count_no_issues(self):
        """Test content with minimal markdown issues."""
        content = """# Heading

Some text.

- List item

```python
code
```"""
        issues = count_markdown_issues(content)
        # Returns a dict of issue counts
        self.assertIsInstance(issues, dict)
        self.assertEqual(issues['missing_language_tags'], 0)
        self.assertEqual(issues['long_lines'], 0)

    def test_count_code_block_without_language(self):
        """Test detecting code blocks without language."""
        content = """```
code without language
```"""
        issues = count_markdown_issues(content)
        # Returns dict with counts
        self.assertGreater(issues['missing_language_tags'], 0)

    def test_count_multiple_blank_lines(self):
        """Test detecting multiple consecutive blank lines."""
        content = """# Heading


Multiple blank lines above"""
        issues = count_markdown_issues(content)
        # Returns dict with counts
        self.assertGreater(issues['multiple_blank_lines'], 0)


if __name__ == '__main__':
    unittest.main()
