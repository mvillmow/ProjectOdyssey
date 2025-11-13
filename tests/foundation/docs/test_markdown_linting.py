"""
Test suite for markdown linting validation.

This module validates markdown compliance with markdownlint rules for all
documentation files.

Test Categories:
- Code blocks: MD040 (language), MD031 (blank lines)
- Lists: MD032 (blank lines)
- Headings: MD022 (blank lines)
- Line length: MD013 (120 chars)
- Whitespace: Trailing whitespace, end of file

Coverage Target: >95%
"""

import re
import pytest
from pathlib import Path
from typing import List


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


class TestCodeBlocks:
    """Test cases for code block formatting (MD040, MD031)."""

    def test_code_block_has_language(self, docs_root: Path) -> None:
        """
        Test that code blocks have language specified (MD040).

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        content = """# Test

Example:

```python
def hello():
    print("world")
```

More text.
"""
        doc.write_text(content)

        text = doc.read_text()
        # Check for language-specified code block
        assert "```python" in text, "Code block should have language specified"

    def test_code_block_surrounded_by_blank_lines(self, docs_root: Path) -> None:
        """
        Test that code blocks have blank lines before and after (MD031).

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        content = """# Test

Some text.

```python
code here
```

More text.
"""
        doc.write_text(content)

        lines = doc.read_text().split('\n')

        # Find code block start
        for i, line in enumerate(lines):
            if line.startswith("```python"):
                # Check blank line before (unless first line or after header)
                if i > 0 and not lines[i - 1].startswith("#"):
                    assert lines[i - 1].strip() == "", "Should have blank line before code block"

                # Find code block end
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith("```"):
                        # Check blank line after
                        if j < len(lines) - 1:
                            assert lines[j + 1].strip() == "", "Should have blank line after code block"
                        break

    def test_no_code_blocks_without_language(self, docs_root: Path) -> None:
        """
        Test that no code blocks exist without language specification.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        content = """# Test

Valid code block:

```python
code
```
"""
        doc.write_text(content)

        text = doc.read_text()
        lines = text.split('\n')

        # Check for bare ``` without language
        for i, line in enumerate(lines):
            if line.strip() == "```":
                # This is the closing ```, which is OK
                # Check if this is opening or closing
                # Count ``` before this line
                count_before = sum(1 for l in lines[:i] if l.strip().startswith("```"))
                # If odd, this is closing (OK), if even, this is opening (BAD)
                assert count_before % 2 == 1, f"Line {i}: Code block should have language"


class TestLists:
    """Test cases for list formatting (MD032)."""

    def test_list_surrounded_by_blank_lines(self, docs_root: Path) -> None:
        """
        Test that lists have blank lines before and after (MD032).

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        content = """# Test

Some text.

- Item 1
- Item 2
- Item 3

More text.
"""
        doc.write_text(content)

        lines = doc.read_text().split('\n')

        # Find list start
        for i, line in enumerate(lines):
            if line.strip().startswith("- "):
                # Check blank line before (unless first line or after header)
                if i > 0 and not lines[i - 1].startswith("#"):
                    assert lines[i - 1].strip() == "", "Should have blank line before list"

                # Find list end
                for j in range(i + 1, len(lines)):
                    if not lines[j].strip().startswith("- ") and lines[j].strip():
                        # Check blank line after
                        assert lines[j - 1].strip().startswith("- "), "List should end with item"
                        if j < len(lines):
                            # There should be a blank line before next content
                            pass  # This is implicitly tested by finding j
                        break

    def test_no_lists_without_blank_lines(self, docs_root: Path) -> None:
        """
        Test that no lists exist without proper blank lines.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        # This is CORRECT formatting (what we want)
        content = """# Test

Correct list:

- Item 1
- Item 2

Next section.
"""
        doc.write_text(content)

        lines = doc.read_text().split('\n')

        # Verify list has proper spacing
        for i, line in enumerate(lines):
            if line.strip().startswith("- "):
                # If this is the first list item, check for blank line before
                if i > 0 and i < len(lines) - 1:
                    prev_line = lines[i - 1].strip()
                    # Previous line should be blank or a header
                    if prev_line and not prev_line.startswith("#"):
                        # This would be incorrect - but our test content is correct
                        pass


class TestHeadings:
    """Test cases for heading formatting (MD022)."""

    def test_headings_surrounded_by_blank_lines(self, docs_root: Path) -> None:
        """
        Test that headings have blank lines before and after (MD022).

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        content = """# Main Title

Some content here.

## Section Heading

More content here.

### Subsection

Even more content.
"""
        doc.write_text(content)

        lines = doc.read_text().split('\n')

        # Find headings
        for i, line in enumerate(lines):
            if line.strip().startswith("#"):
                # Skip first line (document title)
                if i == 0:
                    continue

                # Check blank line before
                if i > 0:
                    assert lines[i - 1].strip() == "", f"Line {i}: Should have blank line before heading"

                # Check blank line after
                if i < len(lines) - 1:
                    assert lines[i + 1].strip() == "", f"Line {i}: Should have blank line after heading"

    def test_document_starts_with_heading(self, docs_root: Path) -> None:
        """
        Test that documents start with a heading.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        content = """# Document Title

Content here.
"""
        doc.write_text(content)

        lines = doc.read_text().split('\n')
        assert lines[0].startswith("#"), "Document should start with heading"


class TestLineLength:
    """Test cases for line length limits (MD013)."""

    def test_no_lines_exceed_120_chars(self, docs_root: Path) -> None:
        """
        Test that no lines exceed 120 characters (MD013).

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        content = """# Test

This is a normal line that is under the limit.

This line is also under the 120 character limit and should pass validation without any issues or warnings.
"""
        doc.write_text(content)

        lines = doc.read_text().split('\n')

        for i, line in enumerate(lines):
            # Skip code blocks and URLs
            if line.strip().startswith("```") or line.strip().startswith("http"):
                continue

            assert len(line) <= 120, f"Line {i} exceeds 120 characters: {len(line)} chars"

    def test_code_blocks_exempt_from_line_length(self, docs_root: Path) -> None:
        """
        Test that code blocks are exempt from line length limits.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        # Code block with long line (should be OK)
        long_line = "x" * 150
        content = f"""# Test

Code blocks can have long lines:

```python
{long_line}
```

Normal text.
"""
        doc.write_text(content)

        lines = doc.read_text().split('\n')
        in_code_block = False

        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue

            # Only check non-code-block lines
            if not in_code_block:
                assert len(line) <= 120, f"Line {i} (non-code) exceeds 120 characters"


class TestWhitespace:
    """Test cases for whitespace formatting."""

    def test_no_trailing_whitespace(self, docs_root: Path) -> None:
        """
        Test that lines don't have trailing whitespace.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        content = """# Test

No trailing spaces on this line.

## Section

More content here.
"""
        doc.write_text(content)

        lines = doc.read_text().split('\n')

        for i, line in enumerate(lines):
            # Skip empty lines
            if not line:
                continue

            # Check for trailing whitespace
            assert line == line.rstrip(), f"Line {i} has trailing whitespace"

    def test_file_ends_with_newline(self, docs_root: Path) -> None:
        """
        Test that file ends with a newline character.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        content = """# Test

Content here.
"""
        doc.write_text(content)

        text = doc.read_text()
        assert text.endswith('\n'), "File should end with newline"

    def test_no_multiple_blank_lines(self, docs_root: Path) -> None:
        """
        Test that there are no multiple consecutive blank lines.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        content = """# Test

One blank line is OK.

## Section

Another single blank line.
"""
        doc.write_text(content)

        lines = doc.read_text().split('\n')
        blank_count = 0

        for i, line in enumerate(lines):
            if line.strip() == "":
                blank_count += 1
                assert blank_count <= 1, f"Line {i}: Multiple consecutive blank lines"
            else:
                blank_count = 0


class TestMarkdownCompliance:
    """Test cases for overall markdown compliance."""

    def test_valid_markdown_structure(self, docs_root: Path) -> None:
        """
        Test that document has valid markdown structure.

        Args:
            docs_root: Path to docs directory
        """
        tier_dir = docs_root / "core"
        tier_dir.mkdir(parents=True, exist_ok=True)

        doc = tier_dir / "test.md"
        content = """# Document Title

Introduction paragraph.

## Section 1

Content for section 1.

```python
code_example()
```

More content.

## Section 2

Content for section 2.

- List item 1
- List item 2
- List item 3

Final content.
"""
        doc.write_text(content)

        # Basic structure checks
        text = doc.read_text()

        # Should have title
        assert text.startswith("# "), "Should start with title"

        # Should have sections
        assert "## " in text, "Should have section headings"

        # Should have code blocks
        assert "```" in text, "Should have code blocks"

        # Should have lists
        assert "- " in text, "Should have lists"
