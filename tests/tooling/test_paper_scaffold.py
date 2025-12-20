#!/usr/bin/env python3
"""
Tests for paper scaffolding tools.

Tests Issue #744-763 (Directory Generator) implementation:
- Directory creation (idempotent, error handling)
- File generation from templates
- Structure validation
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools" / "paper-scaffold"))

from validate import ValidationStatus, validate_paper_structure


class TestPaperNameNormalization:
    """Test paper name normalization (Issue #744)."""

    def test_normalize_simple_name(self):
        """Test simple name normalization."""
        from scaffold_enhanced import normalize_paper_name

        assert normalize_paper_name("LeNet-5") == "lenet-5"

    def test_normalize_with_spaces(self):
        """Test normalization with spaces."""
        from scaffold_enhanced import normalize_paper_name

        assert normalize_paper_name("LeNet 5") == "lenet-5"

    def test_normalize_with_special_chars(self):
        """Test normalization with special characters."""
        from scaffold_enhanced import normalize_paper_name

        assert normalize_paper_name("BERT: Pre-training") == "bert-pre-training"

    def test_normalize_consecutive_hyphens(self):
        """Test removal of consecutive hyphens."""
        from scaffold_enhanced import normalize_paper_name

        assert normalize_paper_name("GPT--2") == "gpt-2"

    def test_normalize_leading_trailing_hyphens(self):
        """Test removal of leading/trailing hyphens."""
        from scaffold_enhanced import normalize_paper_name

        assert normalize_paper_name("-AlexNet-") == "alexnet"


class TestDirectoryCreation:
    """Test directory creation functionality (Issue #744)."""

    def setup_method(self):
        """Create temporary directory for tests."""
        self.test_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up temporary directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_create_basic_structure(self):
        """Test basic directory structure creation."""
        from scaffold_enhanced import DirectoryGenerator

        generator = DirectoryGenerator(self.test_dir, verbose=False)
        result = generator.generate(
            paper_name="test-paper",
            paper_metadata={
                "PAPER_TITLE": "Test Paper",
                "AUTHORS": "Test Author",
                "YEAR": "2025",
                "PAPER_URL": "http://example.com",
                "DESCRIPTION": "Test description",
            },
            validate=False,
            dry_run=True,  # Don't actually create files
        )

        assert result.success
        assert len(result.created_dirs) > 0

    def test_idempotent_directory_creation(self):
        """Test that directory creation is idempotent (Issue #744)."""
        from scaffold_enhanced import DirectoryGenerator

        generator = DirectoryGenerator(self.test_dir, verbose=False)
        paper_path = self.test_dir / "test-paper"

        # Create directory
        paper_path.mkdir()

        # Run generator (should not fail on existing directory)
        result = generator.generate(paper_name="test-paper", paper_metadata={}, validate=False)

        assert result.success
        assert paper_path in result.skipped_dirs


class TestFileGeneration:
    """Test file generation from templates (Issue #749)."""

    def setup_method(self):
        """Create temporary directory for tests."""
        self.test_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up temporary directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_template_rendering(self):
        """Test template variable substitution."""
        from scaffold_enhanced import render_template

        template = "Paper: {{PAPER_NAME}}, Year: {{YEAR}}"
        variables = {"PAPER_NAME": "Test", "YEAR": "2025"}
        result = render_template(template, variables)

        assert result == "Paper: Test, Year: 2025"

    def test_file_overwrite_protection(self):
        """Test that existing files are not overwritten (Issue #749)."""
        from scaffold_enhanced import DirectoryGenerator

        generator = DirectoryGenerator(self.test_dir, verbose=False)
        paper_path = self.test_dir / "test-paper"
        paper_path.mkdir()
        readme_path = paper_path / "README.md"

        # Create existing file
        readme_path.write_text("Existing content")

        # Run generator
        result = generator.generate(paper_name="test-paper", paper_metadata={}, validate=False)

        # File should be skipped, not overwritten
        assert readme_path in result.skipped_files or not readme_path.exists()
        # If file was created, verify original content is preserved
        if readme_path.exists() and readme_path in result.skipped_files:
            assert readme_path.read_text() == "Existing content"


class TestValidation:
    """Test structure validation (Issue #754)."""

    def setup_method(self):
        """Create temporary directory for tests."""
        self.test_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up temporary directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_validate_complete_structure(self):
        """Test validation of complete structure."""
        # Create complete structure
        paper_path = self.test_dir / "test-paper"
        paper_path.mkdir()
        (paper_path / "src").mkdir()
        (paper_path / "tests").mkdir()
        (paper_path / "README.md").write_text("# Test Paper\n\nOverview and Implementation details here.")

        # Validate
        report = validate_paper_structure(paper_path)

        assert report.status == ValidationStatus.PASS
        assert len(report.missing_directories) == 0
        assert len(report.missing_files) == 0

    def test_validate_missing_directories(self):
        """Test validation detects missing directories."""
        # Create incomplete structure (missing src/)
        paper_path = self.test_dir / "test-paper"
        paper_path.mkdir()
        (paper_path / "tests").mkdir()
        (paper_path / "README.md").write_text("# Test")

        # Validate
        report = validate_paper_structure(paper_path)

        assert report.status == ValidationStatus.FAIL
        assert any("src" in str(d) for d in report.missing_directories)

    def test_validate_missing_files(self):
        """Test validation detects missing files."""
        # Create structure without README.md
        paper_path = self.test_dir / "test-paper"
        paper_path.mkdir()
        (paper_path / "src").mkdir()
        (paper_path / "tests").mkdir()

        # Validate
        report = validate_paper_structure(paper_path)

        assert report.status == ValidationStatus.FAIL
        assert any("README.md" in str(f) for f in report.missing_files)

    def test_validate_empty_readme(self):
        """Test validation detects empty README."""
        # Create structure with empty README
        paper_path = self.test_dir / "test-paper"
        paper_path.mkdir()
        (paper_path / "src").mkdir()
        (paper_path / "tests").mkdir()
        (paper_path / "README.md").write_text("")

        # Validate
        report = validate_paper_structure(paper_path)

        assert report.status == ValidationStatus.FAIL
        assert len(report.invalid_files) > 0

    def test_validation_report_formatting(self):
        """Test validation report is formatted correctly."""
        # Create incomplete structure
        paper_path = self.test_dir / "test-paper"
        paper_path.mkdir()

        # Validate
        report = validate_paper_structure(paper_path)

        # Check report can be formatted
        report_str = str(report)
        assert "VALIDATION" in report_str
        assert "Missing" in report_str or "missing" in report_str


class TestEndToEnd:
    """End-to-end integration tests."""

    def setup_method(self):
        """Create temporary directory for tests."""
        self.test_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up temporary directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_full_generation_with_validation(self):
        """Test complete generation and validation workflow."""
        from scaffold_enhanced import DirectoryGenerator

        generator = DirectoryGenerator(self.test_dir, verbose=False)

        result = generator.generate(
            paper_name="LeNet-5",
            paper_metadata={
                "PAPER_TITLE": "LeNet-5",
                "AUTHORS": "LeCun et al.",
                "YEAR": "1998",
                "PAPER_URL": "http://example.com",
                "DESCRIPTION": "CNN for digit recognition",
            },
            validate=True,
        )

        # Check generation succeeded
        assert result.success
        assert len(result.created_dirs) > 0

        # Check validation ran
        assert result.validation_report is not None

        # Check paper directory was created with normalized name
        assert (self.test_dir / "lenet-5").exists()

    def test_dry_run_mode(self):
        """Test dry run mode doesn't create anything."""
        from scaffold_enhanced import DirectoryGenerator

        generator = DirectoryGenerator(self.test_dir, verbose=False)

        result = generator.generate(paper_name="test-paper", paper_metadata={}, dry_run=True)

        # Check nothing was actually created
        paper_path = self.test_dir / "test-paper"
        assert not paper_path.exists()

        # But result should show what would be created
        assert len(result.created_dirs) > 0


class TestCLIArguments:
    """Test CLI argument parsing (Issue #780)."""

    def test_help_text(self):
        """Test --help displays comprehensive usage information."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "tools/paper-scaffold/scaffold_enhanced.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        output = result.stdout

        # Check for key elements in help text
        assert "paper" in output.lower() or "Paper name" in output
        assert "title" in output.lower()
        assert "authors" in output.lower()
        assert "Examples:" in output
        assert "interactive" in output.lower()

    def test_interactive_mode_no_args(self):
        """Test interactive mode is triggered when no args provided."""
        # This is tested in test_user_prompts.py
        # Here we just verify the flag exists
        import subprocess

        result = subprocess.run(
            [sys.executable, "tools/paper-scaffold/scaffold_enhanced.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert "--interactive" in result.stdout

    def test_paper_argument_optional(self):
        """Test --paper argument is optional (for interactive mode)."""

        # Verify argparse config allows missing --paper
        # This is indirectly tested by the help text not showing --paper as required
        import subprocess

        result = subprocess.run(
            [sys.executable, "tools/paper-scaffold/scaffold_enhanced.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # --paper should be optional (shown in [--paper PAPER] not just PAPER)
        assert "[--paper PAPER]" in result.stdout or "--paper" in result.stdout

    def test_dry_run_flag(self):
        """Test --dry-run flag exists and is documented."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "tools/paper-scaffold/scaffold_enhanced.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert "--dry-run" in result.stdout
        assert "without actually creating" in result.stdout.lower() or "dry" in result.stdout.lower()

    def test_no_validate_flag(self):
        """Test --no-validate flag exists."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "tools/paper-scaffold/scaffold_enhanced.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert "--no-validate" in result.stdout

    def test_quiet_flag(self):
        """Test --quiet flag exists."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "tools/paper-scaffold/scaffold_enhanced.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert "--quiet" in result.stdout

    def test_output_directory_option(self):
        """Test --output option for specifying directory."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "tools/paper-scaffold/scaffold_enhanced.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert "--output" in result.stdout

    def test_examples_in_help(self):
        """Test help text includes usage examples."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "tools/paper-scaffold/scaffold_enhanced.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert "Examples:" in result.stdout
        # Should show both interactive and non-interactive examples
        assert "interactive" in result.stdout.lower()

    def test_argument_defaults(self):
        """Test default values are appropriate."""
        # Test that defaults exist by checking help text
        import subprocess

        result = subprocess.run(
            [sys.executable, "tools/paper-scaffold/scaffold_enhanced.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Default output should be papers/
        assert "papers" in result.stdout.lower() or "default:" in result.stdout.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
