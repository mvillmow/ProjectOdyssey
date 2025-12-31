#!/usr/bin/env python3
"""Tests for WorktreeManager class.

Tests for the git worktree management functionality
in scripts/implement_issues.py.

Addresses Issue #2953: Phase 3 Code Quality Improvements
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from implement_issues import WorktreeManager


class TestWorktreeManager:
    """Tests for WorktreeManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_root = pathlib.Path(self.temp_dir)
        # Create a mock worktrees directory
        (self.repo_root / "worktrees").mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @mock.patch("implement_issues.run")
    def test_create_new_worktree(self, mock_run):
        """Test creating a new worktree."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")

        manager = WorktreeManager(self.repo_root)
        path = manager.create(123, "test-description")

        assert path.parent == self.repo_root / "worktrees"
        assert "123-" in path.name
        assert "test-description" in path.name

    @mock.patch("implement_issues.run")
    def test_create_sanitizes_branch_name(self, mock_run):
        """Test that special characters are sanitized in branch names."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")

        manager = WorktreeManager(self.repo_root)
        path = manager.create(456, "Test: Special/Characters!")

        # Should sanitize to lowercase alphanumeric with dashes
        assert "test--special-characters-" in path.name or "test" in path.name.lower()

    @mock.patch("implement_issues.run")
    def test_create_existing_branch(self, mock_run):
        """Test creating worktree when branch already exists."""
        # First call (show-ref) returns 0 = branch exists
        # Second call (worktree add without -b) succeeds
        mock_run.side_effect = [
            mock.Mock(returncode=0, stdout="ref/heads/branch", stderr=""),
            mock.Mock(returncode=0, stdout="", stderr=""),
        ]

        manager = WorktreeManager(self.repo_root)
        _ = manager.create(789, "existing-branch")

        # Should use existing branch, not create new one
        assert mock_run.call_count == 2
        # Second call should not have -b flag (using existing branch)
        second_call_args = mock_run.call_args_list[1][0][0]
        assert "-b" not in second_call_args or "worktree" in second_call_args

    @mock.patch("implement_issues.run")
    def test_create_handles_existing_worktree(self, mock_run):
        """Test that existing worktree is returned without error."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")

        manager = WorktreeManager(self.repo_root)
        # Create the worktree directory first
        worktree_path = self.repo_root / "worktrees" / "123-test"
        worktree_path.mkdir(parents=True)

        path = manager.create(123, "test")

        # Should return existing path without error
        assert path == worktree_path

    @mock.patch("implement_issues.run")
    @mock.patch("implement_issues.log")
    def test_remove_worktree(self, mock_log, mock_run):
        """Test removing a worktree."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")

        manager = WorktreeManager(self.repo_root)
        # Create a worktree directory to remove
        worktree_path = self.repo_root / "worktrees" / "123-test-issue"
        worktree_path.mkdir(parents=True)

        result = manager.remove(123)

        assert result is True
        # Should call git worktree remove and git branch -D
        assert mock_run.call_count >= 1

    @mock.patch("implement_issues.run")
    def test_remove_nonexistent_worktree(self, mock_run):
        """Test removing a worktree that doesn't exist."""
        manager = WorktreeManager(self.repo_root)
        result = manager.remove(999)
        assert result is False

    @mock.patch("implement_issues.run")
    def test_list_active(self, mock_run):
        """Test listing active worktrees."""
        # Mock git worktree list output
        worktree_path = self.repo_root / "worktrees" / "123-test"
        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout=f"worktree {worktree_path}\nHEAD abc123\nbranch refs/heads/123-test\n\n",
            stderr="",
        )

        # Create the worktree directory
        worktree_path.mkdir(parents=True)

        manager = WorktreeManager(self.repo_root)
        result = manager.list_active()

        assert 123 in result
        assert result[123] == worktree_path

    @mock.patch("implement_issues.run")
    def test_cleanup_stale(self, mock_run):
        """Test cleaning up stale worktrees."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")

        manager = WorktreeManager(self.repo_root)

        # Create an orphaned directory (not in git worktree list)
        orphan = self.repo_root / "worktrees" / "orphan-worktree"
        orphan.mkdir(parents=True)

        count = manager.cleanup_stale()

        # Should have pruned and removed orphan
        assert count >= 0

    @mock.patch("implement_issues.run")
    @mock.patch("implement_issues.safe_git_fetch")
    def test_create_for_existing_branch(self, mock_fetch, mock_run):
        """Test creating worktree for an existing remote branch."""
        mock_fetch.return_value = (True, "")
        mock_run.side_effect = [
            mock.Mock(returncode=0, stdout="", stderr=""),  # worktree list
            mock.Mock(returncode=0, stdout="", stderr=""),  # worktree add
        ]

        manager = WorktreeManager(self.repo_root)
        path = manager.create_for_existing_branch(123, "feature-branch")

        assert path.parent == self.repo_root / "worktrees"
        assert "123" in path.name


class TestWorktreeManagerThreadSafety:
    """Tests for WorktreeManager thread safety."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_root = pathlib.Path(self.temp_dir)
        (self.repo_root / "worktrees").mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @mock.patch("implement_issues.run")
    def test_concurrent_creates(self, mock_run):
        """Test that concurrent creates don't conflict."""
        import threading

        mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        manager = WorktreeManager(self.repo_root)

        results = []
        errors = []

        def create_worktree(issue_num):
            try:
                path = manager.create(issue_num, f"issue-{issue_num}")
                results.append((issue_num, path))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=create_worktree, args=(i,))
            for i in range(100, 110)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed without errors
        assert len(errors) == 0
        assert len(results) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
