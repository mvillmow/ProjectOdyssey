#!/usr/bin/env python3
"""
Script validation tests for agent utility scripts.

Tests that validation scripts work correctly, setup scripts handle errors,
and utility scripts produce expected output.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest


# ============================================================================
# Path Helpers
# ============================================================================


@pytest.fixture(scope="session")
def scripts_dir(repo_root: Path) -> Path:
    """Get scripts/agents directory."""
    return repo_root / "scripts" / "agents"


@pytest.fixture(scope="session")
def agent_scripts(scripts_dir: Path) -> List[Path]:
    """Get all agent utility scripts."""
    if not scripts_dir.exists():
        return []
    return sorted(
        [f for f in scripts_dir.glob("*.py") if f.name not in ["__init__.py"] and not f.name.startswith("test_")]
    )


# ============================================================================
# Script Execution Helpers
# ============================================================================


def run_script(
    script_path: Path, args: List[str] = None, cwd: Path = None, check: bool = False
) -> subprocess.CompletedProcess:
    """
    Run a Python script and return the result.

    Args:
        script_path: Path to the script
        args: Command-line arguments
        cwd: Working directory
        check: Whether to raise on non-zero exit

    Returns:
        CompletedProcess with stdout, stderr, and returncode
    """
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    return subprocess.run(cmd, cwd=cwd or script_path.parent.parent.parent, capture_output=True, text=True, check=check)


# ============================================================================
# Script Existence Tests
# ============================================================================


@pytest.mark.scripts
class TestScriptsExist:
    """Test that all expected utility scripts exist."""

    def test_scripts_directory_exists(self, scripts_dir: Path):
        """Test that scripts/agents directory exists."""
        assert scripts_dir.exists(), f"Scripts directory not found: {scripts_dir}"

    def test_expected_scripts_exist(self, scripts_dir: Path):
        """Test that expected utility scripts exist."""
        expected_scripts = [
            "validate_agents.py",
            "list_agents.py",
            "agent_stats.py",
            "check_frontmatter.py",
            "test_agent_loading.py",
        ]

        for script_name in expected_scripts:
            script_path = scripts_dir / script_name
            assert script_path.exists(), f"Expected script not found: {script_name}"

    def test_script_is_executable(self, agent_scripts: List[Path]):
        """Test that each script has executable permissions or shebang."""
        for script_file in agent_scripts:
            content = script_file.read_text()

            # Should have shebang
            assert content.startswith("#!"), f"Script {script_file.name} missing shebang line"

            # Shebang should be python3
            first_line = content.split("\n")[0]
            assert "python" in first_line.lower(), f"Script {script_file.name} shebang not for Python: {first_line}"

    def test_script_has_docstring(self, agent_scripts: List[Path]):
        """Test that each script has a module docstring."""
        for script_file in agent_scripts:
            content = script_file.read_text()

            # Look for docstring after shebang
            lines = content.split("\n")
            doc_start = None
            for i, line in enumerate(lines[1:10], 1):  # Check first 10 lines
                if '"""' in line or "'''" in line:
                    doc_start = i
                    break

            assert doc_start is not None, f"Script {script_file.name} missing module docstring"


# ============================================================================
# validate_agents.py Tests
# ============================================================================


@pytest.mark.scripts
class TestValidateAgentsScript:
    """Test validate_agents.py script."""

    def test_help_option(self, scripts_dir: Path, repo_root: Path):
        """Test that --help option works."""
        script = scripts_dir / "validate_agents.py"
        result = run_script(script, ["--help"], cwd=repo_root)

        assert result.returncode == 0, "Help option should succeed"
        assert "usage:" in result.stdout.lower(), "Help should show usage"
        assert "--verbose" in result.stdout, "Help should mention verbose option"

    def test_runs_without_errors(self, scripts_dir: Path, repo_root: Path, agents_dir: Path):
        """Test that validate_agents.py runs without errors."""
        if not agents_dir.exists() or not list(agents_dir.glob("*.md")):
            pytest.skip("No agent files to validate")

        script = scripts_dir / "validate_agents.py"
        result = run_script(script, cwd=repo_root)

        # Should exit with 0 or 1 (not crash)
        assert result.returncode in [0, 1], f"Script crashed with code {result.returncode}: {result.stderr}"

    def test_verbose_option_works(self, scripts_dir: Path, repo_root: Path, agents_dir: Path):
        """Test that --verbose option produces more output."""
        if not agents_dir.exists() or not list(agents_dir.glob("*.md")):
            pytest.skip("No agent files to validate")

        script = scripts_dir / "validate_agents.py"

        # Run without verbose
        result_normal = run_script(script, cwd=repo_root)

        # Run with verbose
        result_verbose = run_script(script, ["--verbose"], cwd=repo_root)

        # Verbose should have more or equal output
        assert len(result_verbose.stdout) >= len(result_normal.stdout), (
            "Verbose mode should produce at least as much output"
        )

    def test_detects_missing_frontmatter(self, scripts_dir: Path, tmp_path: Path):
        """Test that script detects missing frontmatter."""
        # Create a test file without frontmatter
        test_agents_dir = tmp_path / "agents"
        test_agents_dir.mkdir()

        test_file = test_agents_dir / "test-agent.md"
        test_file.write_text("# Test Agent\n\nNo frontmatter here.")

        script = scripts_dir / "validate_agents.py"
        result = run_script(script, ["--agents-dir", str(test_agents_dir)], cwd=tmp_path)

        # Should fail validation
        assert result.returncode == 1, "Should detect missing frontmatter"
        assert "frontmatter" in result.stdout.lower() or "frontmatter" in result.stderr.lower()


# ============================================================================
# list_agents.py Tests
# ============================================================================


@pytest.mark.scripts
class TestListAgentsScript:
    """Test list_agents.py script."""

    def test_help_option(self, scripts_dir: Path, repo_root: Path):
        """Test that --help option works."""
        script = scripts_dir / "list_agents.py"
        result = run_script(script, ["--help"], cwd=repo_root)

        assert result.returncode == 0, "Help option should succeed"
        assert "usage:" in result.stdout.lower()
        assert "--level" in result.stdout, "Help should mention level filter"
        assert "--verbose" in result.stdout, "Help should mention verbose option"

    def test_lists_agents(self, scripts_dir: Path, repo_root: Path, agents_dir: Path):
        """Test that list_agents.py lists agents."""
        if not agents_dir.exists() or not list(agents_dir.glob("*.md")):
            pytest.skip("No agent files to list")

        script = scripts_dir / "list_agents.py"
        result = run_script(script, cwd=repo_root)

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "Total Agents:" in result.stdout, "Should show agent count"

    def test_level_filter_works(self, scripts_dir: Path, repo_root: Path, agents_dir: Path):
        """Test that --level filter works."""
        if not agents_dir.exists() or not list(agents_dir.glob("*.md")):
            pytest.skip("No agent files to list")

        script = scripts_dir / "list_agents.py"

        # Try filtering by level 0
        result = run_script(script, ["--level", "0"], cwd=repo_root)

        # Should succeed and mention Level 0
        if result.returncode == 0:
            assert "Level 0" in result.stdout, "Should show Level 0 section"

    def test_verbose_shows_details(self, scripts_dir: Path, repo_root: Path, agents_dir: Path):
        """Test that --verbose shows agent details."""
        if not agents_dir.exists() or not list(agents_dir.glob("*.md")):
            pytest.skip("No agent files to list")

        script = scripts_dir / "list_agents.py"
        result = run_script(script, ["--verbose"], cwd=repo_root)

        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Verbose mode should show tools or model info
        output_lower = result.stdout.lower()
        has_details = any(keyword in output_lower for keyword in ["tools:", "model:", "file:"])
        assert has_details, "Verbose mode should show agent details"


# ============================================================================
# agent_stats.py Tests
# ============================================================================


@pytest.mark.scripts
class TestAgentStatsScript:
    """Test agent_stats.py script."""

    def test_help_option(self, scripts_dir: Path, repo_root: Path):
        """Test that --help option works."""
        script = scripts_dir / "agent_stats.py"
        result = run_script(script, ["--help"], cwd=repo_root)

        assert result.returncode == 0, "Help option should succeed"
        assert "--format" in result.stdout, "Help should mention format option"

    def test_text_format(self, scripts_dir: Path, repo_root: Path, agents_dir: Path):
        """Test text format output."""
        if not agents_dir.exists() or not list(agents_dir.glob("*.md")):
            pytest.skip("No agent files for stats")

        script = scripts_dir / "agent_stats.py"
        result = run_script(script, ["--format", "text"], cwd=repo_root)

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "Total Agents:" in result.stdout, "Should show total agents"

    def test_json_format(self, scripts_dir: Path, repo_root: Path, agents_dir: Path):
        """Test JSON format output."""
        if not agents_dir.exists() or not list(agents_dir.glob("*.md")):
            pytest.skip("No agent files for stats")

        script = scripts_dir / "agent_stats.py"
        result = run_script(script, ["--format", "json"], cwd=repo_root)

        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Should be valid JSON
        try:
            data = json.loads(result.stdout)
            assert "total_agents" in data, "JSON should have total_agents field"
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")

    def test_markdown_format(self, scripts_dir: Path, repo_root: Path, agents_dir: Path):
        """Test markdown format output."""
        if not agents_dir.exists() or not list(agents_dir.glob("*.md")):
            pytest.skip("No agent files for stats")

        script = scripts_dir / "agent_stats.py"
        result = run_script(script, ["--format", "markdown"], cwd=repo_root)

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "# Agent System Statistics" in result.stdout, "Markdown should have main heading"
        assert "## " in result.stdout, "Markdown should have section headings"

    def test_output_file_option(self, scripts_dir: Path, repo_root: Path, tmp_path: Path, agents_dir: Path):
        """Test writing output to file."""
        if not agents_dir.exists() or not list(agents_dir.glob("*.md")):
            pytest.skip("No agent files for stats")

        output_file = tmp_path / "stats.txt"
        script = scripts_dir / "agent_stats.py"

        result = run_script(script, ["--format", "text", "--output", str(output_file)], cwd=repo_root)

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert output_file.exists(), "Output file should be created"
        assert len(output_file.read_text()) > 0, "Output file should have content"


# ============================================================================
# check_frontmatter.py Tests
# ============================================================================


@pytest.mark.scripts
class TestCheckFrontmatterScript:
    """Test check_frontmatter.py script."""

    def test_help_option(self, scripts_dir: Path, repo_root: Path):
        """Test that --help option works."""
        script = scripts_dir / "check_frontmatter.py"
        result = run_script(script, ["--help"], cwd=repo_root)

        assert result.returncode == 0, "Help option should succeed"
        assert "usage:" in result.stdout.lower()

    def test_validates_frontmatter(self, scripts_dir: Path, repo_root: Path, agents_dir: Path):
        """Test that check_frontmatter.py validates frontmatter."""
        if not agents_dir.exists() or not list(agents_dir.glob("*.md")):
            pytest.skip("No agent files to check")

        script = scripts_dir / "check_frontmatter.py"
        result = run_script(script, cwd=repo_root)

        # Should complete (exit 0 or 1, not crash)
        assert result.returncode in [0, 1], f"Script crashed with code {result.returncode}"


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.scripts
class TestScriptErrorHandling:
    """Test that scripts handle error conditions gracefully."""

    def test_missing_agents_dir(self, scripts_dir: Path, tmp_path: Path):
        """Test that scripts handle missing agents directory."""
        # Run scripts with a non-existent agents directory
        nonexistent_dir = tmp_path / "nonexistent"

        scripts_to_test = ["validate_agents.py", "list_agents.py"]

        for script_name in scripts_to_test:
            script = scripts_dir / script_name
            if not script.exists():
                continue

            result = run_script(script, ["--agents-dir", str(nonexistent_dir)], cwd=tmp_path)

            # Should fail gracefully with error message
            assert result.returncode != 0, f"{script_name} should fail with missing directory"
            assert len(result.stderr) > 0 or "Error" in result.stdout, f"{script_name} should show error message"

    def test_empty_agents_dir(self, scripts_dir: Path, tmp_path: Path):
        """Test that scripts handle empty agents directory."""
        # Create empty directory
        empty_dir = tmp_path / "empty_agents"
        empty_dir.mkdir()

        scripts_to_test = ["validate_agents.py", "list_agents.py", "agent_stats.py"]

        for script_name in scripts_to_test:
            script = scripts_dir / script_name
            if not script.exists():
                continue

            result = run_script(script, ["--agents-dir", str(empty_dir)], cwd=tmp_path)

            # Should handle gracefully (may fail or show empty results)
            assert result.returncode in [0, 1], f"{script_name} should handle empty directory gracefully"

    def test_invalid_arguments(self, scripts_dir: Path, repo_root: Path):
        """Test that scripts reject invalid arguments."""
        script = scripts_dir / "list_agents.py"

        # Try invalid level
        result = run_script(script, ["--level", "99"], cwd=repo_root)

        # Should fail or show error
        assert result.returncode != 0, "Should reject invalid level"


# ============================================================================
# Script Output Format Tests
# ============================================================================


@pytest.mark.scripts
class TestScriptOutputFormats:
    """Test that script outputs are well-formatted."""

    def test_validate_agents_summary(self, scripts_dir: Path, repo_root: Path, agents_dir: Path):
        """Test that validate_agents.py shows summary."""
        if not agents_dir.exists() or not list(agents_dir.glob("*.md")):
            pytest.skip("No agent files to validate")

        script = scripts_dir / "validate_agents.py"
        result = run_script(script, cwd=repo_root)

        output = result.stdout + result.stderr

        # Should show summary statistics
        summary_indicators = ["Total files:", "Total agents:", "files", "agents"]

        has_summary = any(indicator in output for indicator in summary_indicators)
        assert has_summary, "Should show summary of validation results"

    def test_list_agents_readable_output(self, scripts_dir: Path, repo_root: Path, agents_dir: Path):
        """Test that list_agents.py output is human-readable."""
        if not agents_dir.exists() or not list(agents_dir.glob("*.md")):
            pytest.skip("No agent files to list")

        script = scripts_dir / "list_agents.py"
        result = run_script(script, cwd=repo_root)

        assert result.returncode == 0, "Script should succeed"

        # Output should have some structure
        lines = result.stdout.split("\n")
        assert len(lines) > 3, "Should have multiple lines of output"

        # Should have some organizational elements
        has_structure = any(c in result.stdout for c in ["=", "-", "Level", ":", "\n\n"])
        assert has_structure, "Output should have organizational structure"


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.scripts
@pytest.mark.slow
class TestScriptIntegration:
    """Test integration between scripts."""

    def test_all_scripts_use_same_agents_dir(self, scripts_dir: Path, repo_root: Path, agents_dir: Path):
        """Test that all scripts find the same agents directory."""
        if not agents_dir.exists():
            pytest.skip("No agents directory")

        scripts_with_agents_dir = ["validate_agents.py", "list_agents.py", "agent_stats.py"]

        agent_counts = {}

        for script_name in scripts_with_agents_dir:
            script = scripts_dir / script_name
            if not script.exists():
                continue

            result = run_script(script, cwd=repo_root)

            if result.returncode == 0:
                output = result.stdout

                # Extract agent count from output
                # Different scripts format this differently
                import re

                count_patterns = [
                    r"Total [Aa]gents?:\s*(\d+)",
                    r"(\d+)\s+agents?",
                    r'"total_agents":\s*(\d+)',
                ]

                for pattern in count_patterns:
                    match = re.search(pattern, output)
                    if match:
                        agent_counts[script_name] = int(match.group(1))
                        break

        # If we got counts from multiple scripts, they should match
        if len(agent_counts) > 1:
            counts = list(agent_counts.values())
            assert all(c == counts[0] for c in counts), f"Scripts report different agent counts: {agent_counts}"

    def test_validate_then_list(self, scripts_dir: Path, repo_root: Path, agents_dir: Path):
        """Test that validate and list work on same agents."""
        if not agents_dir.exists() or not list(agents_dir.glob("*.md")):
            pytest.skip("No agent files")

        # First validate
        validate_script = scripts_dir / "validate_agents.py"
        validate_result = run_script(validate_script, cwd=repo_root)

        # Then list
        list_script = scripts_dir / "list_agents.py"
        list_result = run_script(list_script, cwd=repo_root)

        # Both should succeed or gracefully handle issues
        assert validate_result.returncode in [0, 1], "Validate should not crash"
        assert list_result.returncode in [0, 1], "List should not crash"
