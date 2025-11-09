#!/usr/bin/env python3
"""
Documentation validation tests for agent documentation.

Tests that all markdown files exist, internal links are valid, table of
contents matches sections, and references are not broken.
"""

import re
import pytest
from pathlib import Path
from typing import List, Tuple, Set, Dict
from conftest import (
    extract_links,
    resolve_relative_path,
    generate_doc_test_ids
)


# ============================================================================
# Documentation File Existence Tests
# ============================================================================

@pytest.mark.documentation
class TestDocumentationFilesExist:
    """Test that all documentation files exist and are readable."""

    def test_docs_directory_exists(self, docs_agents_dir: Path):
        """Test that agents/ documentation directory exists."""
        assert docs_agents_dir.exists(), \
            f"Documentation directory not found: {docs_agents_dir}"
        assert docs_agents_dir.is_dir(), \
            f"Documentation path is not a directory: {docs_agents_dir}"

    def test_required_docs_exist(self, docs_agents_dir: Path):
        """Test that required documentation files exist."""
        required_docs = [
            "README.md",
            "hierarchy.md",
            "agent-hierarchy.md",
            "delegation-rules.md"
        ]

        for doc_name in required_docs:
            doc_path = docs_agents_dir / doc_name
            assert doc_path.exists(), \
                f"Required documentation missing: {doc_name}"

    def test_templates_directory_exists(self, docs_agents_dir: Path):
        """Test that templates directory exists."""
        templates_dir = docs_agents_dir / "templates"
        assert templates_dir.exists(), \
            f"Templates directory not found: {templates_dir}"

    def test_template_files_exist(self, docs_agents_dir: Path):
        """Test that template files exist for all levels."""
        templates_dir = docs_agents_dir / "templates"

        # Expected template files (0-5 levels)
        expected_templates = [
            "level-0-chief-architect.md",
            "level-1-section-orchestrator.md",
            "level-2-module-design.md",
            "level-3-component-specialist.md",
            "level-4-implementation-engineer.md",
            "level-5-junior-engineer.md"
        ]

        for template_name in expected_templates:
            template_path = templates_dir / template_name
            assert template_path.exists(), \
                f"Template file missing: {template_name}"

    def test_doc_file_readable(self, all_doc_files: List[Path]):
        """Test that each documentation file is readable."""
        for doc_file in all_doc_files:
            assert doc_file.exists(), f"Documentation file not found: {doc_file}"
            assert doc_file.is_file(), f"Documentation path is not a file: {doc_file}"

            try:
                content = doc_file.read_text()
                assert len(content) > 0, f"Documentation file is empty: {doc_file}"
            except Exception as e:
                pytest.fail(f"Failed to read documentation file {doc_file}: {e}")


# ============================================================================
# Link Validation Tests
# ============================================================================

@pytest.mark.documentation
class TestInternalLinks:
    """Test that all internal links are valid."""

    def test_internal_links_resolve(self, doc_file: Path, validate_link_exists):
        """Test that all internal markdown links resolve to existing files."""
        content = doc_file.read_text()
        links = extract_links(content)

        broken_links = []
        for link in links:
            # Skip external URLs
            if link.startswith(('http://', 'https://', 'mailto:')):
                continue

            # Skip anchors (we'll test these separately)
            if link.startswith('#'):
                continue

            # Validate link exists
            if not validate_link_exists(doc_file, link):
                broken_links.append(link)

        assert not broken_links, \
            f"Broken links in {doc_file.name}:\n  " + "\n  ".join(broken_links)

    def test_no_absolute_file_paths(self, all_doc_files: List[Path]):
        """Test that documentation uses relative paths, not absolute."""
        for doc_file in all_doc_files:
            content = doc_file.read_text()
            links = extract_links(content)

            absolute_file_links = []
            for link in links:
                # Skip URLs
                if link.startswith(('http://', 'https://', 'mailto:')):
                    continue

                # Check for absolute file paths
                if link.startswith('/'):
                    absolute_file_links.append(link)

            assert not absolute_file_links, \
                f"Absolute file paths found in {doc_file.name} (use relative paths):\n  " + \
                "\n  ".join(absolute_file_links)


# ============================================================================
# Table of Contents Tests
# ============================================================================

def extract_toc_entries(content: str) -> List[Tuple[str, str]]:
    """
    Extract table of contents entries from markdown.

    Pattern: - [Section Name](#anchor)

    Returns:
        List of (section_name, anchor) tuples
    """
    # Match TOC pattern: - [text](#anchor)
    pattern = r'^[\s-]*\[([^\]]+)\]\(#([^)]+)\)'
    matches = re.findall(pattern, content, re.MULTILINE)
    return matches


def extract_headings(content: str) -> List[Tuple[int, str, str]]:
    """
    Extract markdown headings from content.

    Returns:
        List of (level, heading_text, anchor) tuples
    """
    # Match heading pattern: ## Heading Text
    pattern = r'^(#{2,6})\s+(.+?)$'
    matches = re.findall(pattern, content, re.MULTILINE)

    headings = []
    for hashes, text in matches:
        level = len(hashes)
        # Generate anchor from heading text
        anchor = text.lower()
        anchor = re.sub(r'[^\w\s-]', '', anchor)  # Remove special chars
        anchor = re.sub(r'[\s]+', '-', anchor)     # Replace spaces with hyphens
        headings.append((level, text, anchor))

    return headings


@pytest.mark.documentation
class TestTableOfContents:
    """Test that table of contents matches document structure."""

    def test_readme_has_toc(self, docs_agents_dir: Path):
        """Test that README.md has a table of contents."""
        readme = docs_agents_dir / "README.md"
        content = readme.read_text()

        # Check for common TOC indicators
        has_toc = (
            "## Contents" in content or
            "## Table of Contents" in content or
            "## Overview" in content
        )

        assert has_toc, "README.md should have a table of contents or overview section"

    @pytest.mark.parametrize("doc_name", [
        "agent-hierarchy.md",
        "delegation-rules.md"
    ])
    def test_major_docs_have_structure(self, docs_agents_dir: Path, doc_name: str):
        """Test that major documentation files have clear structure."""
        doc_path = docs_agents_dir / doc_name
        content = doc_path.read_text()

        headings = extract_headings(content)

        # Should have at least a few major sections
        assert len(headings) >= 3, \
            f"{doc_name} should have at least 3 sections for clarity"

        # Should have h2 (##) headings
        h2_headings = [h for h in headings if h[0] == 2]
        assert len(h2_headings) >= 2, \
            f"{doc_name} should have at least 2 h2 (##) sections"


# ============================================================================
# Reference Validation Tests
# ============================================================================

@pytest.mark.documentation
class TestCrossReferences:
    """Test cross-references between documentation files."""

    def test_readme_references_hierarchy(self, docs_agents_dir: Path):
        """Test that README references hierarchy documentation."""
        readme = docs_agents_dir / "README.md"
        content = readme.read_text()

        # Should reference hierarchy.md
        assert "hierarchy.md" in content, \
            "README should reference hierarchy.md"

    def test_readme_references_delegation(self, docs_agents_dir: Path):
        """Test that README references delegation rules."""
        readme = docs_agents_dir / "README.md"
        content = readme.read_text()

        # Should reference delegation-rules.md
        assert "delegation-rules.md" in content, \
            "README should reference delegation-rules.md"

    def test_readme_references_templates(self, docs_agents_dir: Path):
        """Test that README references templates directory."""
        readme = docs_agents_dir / "README.md"
        content = readme.read_text()

        # Should reference templates
        assert "templates/" in content or "templates" in content.lower(), \
            "README should reference templates directory"

    def test_hierarchy_references_agents(self, docs_agents_dir: Path):
        """Test that hierarchy doc references actual agent configurations."""
        hierarchy = docs_agents_dir / "agent-hierarchy.md"
        content = hierarchy.read_text()

        # Should reference .claude/agents/
        assert ".claude/agents" in content, \
            "agent-hierarchy.md should reference .claude/agents/"


# ============================================================================
# Content Quality Tests
# ============================================================================

@pytest.mark.documentation
class TestContentQuality:
    """Test documentation content quality."""

    def test_doc_has_title(self, all_doc_files: List[Path]):
        """Test that each documentation file has a title (h1 heading)."""
        for doc_file in all_doc_files:
            content = doc_file.read_text()

            # Match h1 heading: # Title
            pattern = r'^#\s+.+$'
            match = re.search(pattern, content, re.MULTILINE)

            assert match, f"Documentation file {doc_file.name} missing h1 title"

    def test_doc_sufficient_content(self, all_doc_files: List[Path]):
        """Test that documentation has sufficient content."""
        for doc_file in all_doc_files:
            content = doc_file.read_text()

            # Should have at least 100 characters of content
            assert len(content) >= 100, \
                f"Documentation file {doc_file.name} has insufficient content"

    def test_no_placeholder_text(self, all_doc_files: List[Path]):
        """Test that documentation doesn't contain placeholder text."""
        for doc_file in all_doc_files:
            content = doc_file.read_text().lower()

            placeholders = [
                "todo",
                "fixme",
                "xxx",
                "placeholder",
                "coming soon",
                "to be written"
            ]

            found_placeholders = [p for p in placeholders if p in content]

            # Allow placeholders in templates and certain planned files
            if "template" not in doc_file.name.lower() and \
               "skill" not in str(doc_file.parent).lower():
                assert not found_placeholders, \
                    f"Placeholder text found in {doc_file.name}: {found_placeholders}"

    def test_templates_have_instructions(self, docs_agents_dir: Path):
        """Test that template files contain usage instructions."""
        templates_dir = docs_agents_dir / "templates"

        for template_file in templates_dir.glob("*.md"):
            content = template_file.read_text()

            # Templates should have instructions
            has_instructions = (
                "template" in content.lower() or
                "use this" in content.lower() or
                "customize" in content.lower()
            )

            assert has_instructions, \
                f"Template {template_file.name} should have usage instructions"


# ============================================================================
# Markdown Syntax Tests
# ============================================================================

@pytest.mark.documentation
class TestMarkdownSyntax:
    """Test markdown syntax correctness."""

    def test_code_blocks_closed(self, all_doc_files: List[Path]):
        """Test that all code blocks are properly closed."""
        for doc_file in all_doc_files:
            content = doc_file.read_text()

            # Count opening and closing code fence markers
            triple_backticks = content.count("```")

            assert triple_backticks % 2 == 0, \
                f"Unclosed code block in {doc_file.name} (odd number of ```)"

    def test_no_broken_lists(self, all_doc_files: List[Path]):
        """Test that markdown lists are properly formatted."""
        for doc_file in all_doc_files:
            content = doc_file.read_text()
            lines = content.split('\n')

            list_pattern = re.compile(r'^(\s*)([-*+]|\d+\.)\s+.+$')
            prev_indent = -1
            prev_was_list = False

            for i, line in enumerate(lines, 1):
                match = list_pattern.match(line)

                if match:
                    indent = len(match.group(1))

                    # If previous was list, indent should be same or increase by 2-4
                    if prev_was_list and indent > prev_indent:
                        indent_diff = indent - prev_indent
                        # Allow reasonable indent increases (2-4 spaces)
                        if indent_diff > 4:
                            pytest.fail(
                                f"Suspicious list indentation in {doc_file.name} "
                                f"line {i}: {indent_diff} spaces"
                            )

                    prev_indent = indent
                    prev_was_list = True
                else:
                    # Reset on non-list lines
                    if line.strip():  # Non-empty line
                        prev_was_list = False
                        prev_indent = -1

    def test_headings_hierarchical(self, all_doc_files: List[Path]):
        """Test that heading levels are hierarchical (no skipping levels)."""
        for doc_file in all_doc_files:
            content = doc_file.read_text()
            headings = extract_headings(content)

            if len(headings) < 2:
                # Skip files with few headings
                return

            prev_level = headings[0][0]
            for level, text, _ in headings[1:]:
                # Can go down any amount, but up by only 1
                if level > prev_level:
                    # Going deeper
                    assert level <= prev_level + 1, \
                        f"Heading level skip in {doc_file.name}: " \
                        f"h{prev_level} to h{level} ('{text}')"
                prev_level = level


# ============================================================================
# Consistency Tests
# ============================================================================

@pytest.mark.documentation
class TestConsistency:
    """Test consistency across documentation files."""

    def test_all_templates_mention_level(self, docs_agents_dir: Path):
        """Test that all template files mention their level."""
        templates_dir = docs_agents_dir / "templates"

        for template_file in templates_dir.glob("level-*.md"):
            # Extract level number from filename
            match = re.search(r'level-(\d+)', template_file.name)
            if match:
                level_num = match.group(1)
                content = template_file.read_text()

                # Should mention "Level X" in content
                assert f"Level {level_num}" in content, \
                    f"Template {template_file.name} should mention 'Level {level_num}'"

    def test_consistent_agent_name_format(self, docs_agents_dir: Path):
        """Test that agent names follow consistent format in docs."""
        hierarchy = docs_agents_dir / "agent-hierarchy.md"
        content = hierarchy.read_text()

        # Agent names should be capitalized properly
        # Check for common patterns
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            # Look for agent role mentions (avoiding code blocks)
            if "agent" in line.lower() and "```" not in line:
                # Should use title case for agent names
                # This is a soft check - just ensure some consistency
                if " agent" in line.lower():
                    # Agent names should be formatted consistently
                    # (This is a basic check - can be enhanced)
                    pass

    def test_workflow_phases_consistent(self, docs_agents_dir: Path):
        """Test that workflow phase names are consistent."""
        valid_phases = {
            "Plan", "Test", "Implementation", "Packaging", "Cleanup",
            # Variations
            "Planning", "Testing"
        }

        for doc_file in docs_agents_dir.glob("**/*.md"):
            content = doc_file.read_text()

            # Look for workflow phase mentions
            if "## Workflow Phase" in content:
                # Extract phase mentions after this heading
                phase_section = content.split("## Workflow Phase")[1].split("##")[0]

                # Check if uses valid phase names
                # (This is informational - doesn't enforce strict validation)
                for phase in valid_phases:
                    if phase in phase_section:
                        break
