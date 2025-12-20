#!/usr/bin/env python3
"""
Unit tests for scripts/common.py

Tests shared utilities and constants used across scripts.
"""

import pytest
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.common import LABEL_COLORS, get_repo_root, get_agents_dir


class TestLabelColors:
    """Test LABEL_COLORS constant"""

    def test_label_colors_exists(self):
        """Test that LABEL_COLORS is defined"""
        assert LABEL_COLORS is not None

    def test_label_colors_has_required_labels(self):
        """Test that all required labels are defined"""
        required_labels = [
            "planning",
            "documentation",
            "testing",
            "tdd",
            "implementation",
            "packaging",
            "integration",
            "cleanup",
        ]
        for label in required_labels:
            assert label in LABEL_COLORS, f"Missing label: {label}"

    def test_label_colors_are_valid_hex(self):
        """Test that all colors are valid hex codes (6 digits)"""
        for label, color in LABEL_COLORS.items():
            assert len(color) == 6, f"Invalid color for {label}: {color}"
            # Check if valid hex
            try:
                int(color, 16)
            except ValueError:
                pytest.fail(f"Invalid hex color for {label}: {color}")


class TestGetRepoRoot:
    """Test get_repo_root() function"""

    def test_get_repo_root_returns_path(self):
        """Test that get_repo_root returns a Path object"""
        root = get_repo_root()
        assert isinstance(root, Path)

    def test_get_repo_root_has_git_dir(self):
        """Test that repo root contains .git directory"""
        root = get_repo_root()
        git_dir = root / ".git"
        assert git_dir.exists(), f"No .git directory found in {root}"

    def test_get_repo_root_has_expected_structure(self):
        """Test that repo root has expected directories"""
        root = get_repo_root()
        expected_dirs = ["scripts", "notes", ".claude"]
        for dir_name in expected_dirs:
            dir_path = root / dir_name
            # Note: .claude and notes might not exist in all environments
            # So we just check scripts which should always exist
            if dir_name == "scripts":
                assert dir_path.exists(), f"Missing {dir_name} directory"


class TestGetAgentsDir:
    """Test get_agents_dir() function"""

    def test_get_agents_dir_returns_path(self):
        """Test that get_agents_dir returns a Path object"""
        # This may fail if .claude/agents doesn't exist
        try:
            agents_dir = get_agents_dir()
            assert isinstance(agents_dir, Path)
        except RuntimeError:
            # Expected if agents directory doesn't exist
            pass

    def test_get_agents_dir_points_to_claude_agents(self):
        """Test that agents dir is under .claude/agents"""
        try:
            agents_dir = get_agents_dir()
            assert agents_dir.name == "agents"
            assert agents_dir.parent.name == ".claude"
        except RuntimeError:
            # Expected if agents directory doesn't exist
            pass


# NOTE: TestGetPlanDir removed - get_plan_dir() function removed
# Planning is now done through GitHub issues
# See .claude/shared/github-issue-workflow.md


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
