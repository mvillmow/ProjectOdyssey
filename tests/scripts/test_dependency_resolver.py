#!/usr/bin/env python3
"""Tests for DependencyResolver class.

Tests for the dependency resolution and topological sorting functionality
in scripts/implement_issues.py.

Addresses Issue #2953: Phase 3 Code Quality Improvements
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest import mock

import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

# Import the class to test (we'll mock external dependencies)
from implement_issues import DependencyResolver, IssueInfo, ImplementationState


class TestIssueInfo:
    """Tests for IssueInfo dataclass."""

    def test_to_dict(self):
        """Test IssueInfo serialization."""
        info = IssueInfo(
            number=123,
            title="Test Issue",
            depends_on={100, 101},
            priority="P1",
            status="pending",
        )
        result = info.to_dict()
        assert result["number"] == 123
        assert result["title"] == "Test Issue"
        assert set(result["depends_on"]) == {100, 101}
        assert result["priority"] == "P1"
        assert result["status"] == "pending"

    def test_from_dict(self):
        """Test IssueInfo deserialization."""
        data = {
            "number": 456,
            "title": "Another Issue",
            "depends_on": [200, 201],
            "priority": "P0",
            "status": "in_progress",
        }
        info = IssueInfo.from_dict(data)
        assert info.number == 456
        assert info.title == "Another Issue"
        assert info.depends_on == {200, 201}
        assert info.priority == "P0"
        assert info.status == "in_progress"

    def test_from_dict_defaults(self):
        """Test IssueInfo defaults from minimal dict."""
        data = {"number": 789}
        info = IssueInfo.from_dict(data)
        assert info.number == 789
        assert info.title == ""
        assert info.depends_on == set()
        assert info.priority == "P2"
        assert info.status == "pending"


class TestDependencyResolver:
    """Tests for DependencyResolver class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple dependency graph:
        # 1 -> 2 -> 4
        #   \-> 3 -/
        self.issues = {
            1: IssueInfo(number=1, title="Issue 1", depends_on=set(), priority="P0"),
            2: IssueInfo(number=2, title="Issue 2", depends_on={1}, priority="P1"),
            3: IssueInfo(number=3, title="Issue 3", depends_on={1}, priority="P2"),
            4: IssueInfo(number=4, title="Issue 4", depends_on={2, 3}, priority="P1"),
        }
        self.resolver = DependencyResolver(self.issues)

    def test_get_all_issue_numbers(self):
        """Test getting all issue numbers."""
        assert self.resolver.get_all_issue_numbers() == {1, 2, 3, 4}

    @mock.patch("implement_issues.prefetch_issue_states")
    @mock.patch("implement_issues.is_issue_closed")
    def test_get_ready_issues_initial(self, mock_closed, mock_prefetch):
        """Test getting ready issues when nothing is completed."""
        mock_closed.return_value = True  # External deps closed
        ready = self.resolver.get_ready_issues()
        # Only issue 1 has no dependencies
        assert ready == [1]

    @mock.patch("implement_issues.prefetch_issue_states")
    @mock.patch("implement_issues.is_issue_closed")
    def test_get_ready_issues_after_completion(self, mock_closed, mock_prefetch):
        """Test getting ready issues after issue 1 is completed."""
        mock_closed.return_value = True
        self.resolver.mark_completed(1)
        ready = self.resolver.get_ready_issues()
        # Issues 2 and 3 should be ready (their dep 1 is completed)
        assert set(ready) == {2, 3}
        # Should be sorted by priority (P1 before P2)
        assert ready == [2, 3]

    def test_mark_in_progress(self):
        """Test marking issue as in progress."""
        self.resolver.mark_in_progress(1)
        assert 1 in self.resolver._in_progress
        assert self.issues[1].status == "in_progress"

    def test_mark_completed(self):
        """Test marking issue as completed."""
        self.resolver.mark_in_progress(1)
        self.resolver.mark_completed(1)
        assert 1 in self.resolver._completed
        assert 1 not in self.resolver._in_progress
        assert self.issues[1].status == "completed"

    def test_mark_paused(self):
        """Test marking issue as paused."""
        self.resolver.mark_in_progress(1)
        self.resolver.mark_paused(1)
        assert 1 in self.resolver._paused
        assert 1 not in self.resolver._in_progress
        assert self.issues[1].status == "paused"

    def test_has_cycle_no_cycle(self):
        """Test cycle detection with no cycles."""
        assert not self.resolver.has_cycle()

    def test_has_cycle_with_cycle(self):
        """Test cycle detection with a cycle."""
        # Create a cycle: 1 -> 2 -> 3 -> 1
        cyclic_issues = {
            1: IssueInfo(number=1, depends_on={3}),
            2: IssueInfo(number=2, depends_on={1}),
            3: IssueInfo(number=3, depends_on={2}),
        }
        resolver = DependencyResolver(cyclic_issues)
        assert resolver.has_cycle()

    def test_get_topological_order(self):
        """Test topological ordering."""
        order = self.resolver.get_topological_order()
        # Issue 1 must come before 2 and 3
        # Issues 2 and 3 must come before 4
        assert order.index(1) < order.index(2)
        assert order.index(1) < order.index(3)
        assert order.index(2) < order.index(4)
        assert order.index(3) < order.index(4)

    def test_get_topological_order_priority_sorting(self):
        """Test that topological order respects priority."""
        order = self.resolver.get_topological_order()
        # Issue 1 (P0) should come first
        assert order[0] == 1
        # At the same level, P1 should come before P2
        # Issues 2 (P1) and 3 (P2) are both ready after 1
        idx_2 = order.index(2)
        idx_3 = order.index(3)
        assert idx_2 < idx_3  # P1 before P2

    def test_get_topological_order_cycle_raises(self):
        """Test that cycle detection raises ValueError."""
        cyclic_issues = {
            1: IssueInfo(number=1, depends_on={2}),
            2: IssueInfo(number=2, depends_on={1}),
        }
        resolver = DependencyResolver(cyclic_issues)
        with pytest.raises(ValueError, match="Dependency cycle detected"):
            resolver.get_topological_order()

    def test_thread_safety(self):
        """Test that operations are thread-safe."""
        results = []
        errors = []

        def worker(issue_num):
            try:
                self.resolver.mark_in_progress(issue_num)
                # Small delay to increase chance of race conditions
                import time
                time.sleep(0.01)
                self.resolver.mark_completed(issue_num)
                results.append(issue_num)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in [1, 2, 3, 4]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should complete without errors
        assert len(errors) == 0
        assert len(self.resolver._completed) == 4


class TestImplementationState:
    """Tests for ImplementationState dataclass."""

    def test_state_basic(self):
        """Test that ImplementationState can be created with basic data."""
        # Create a state with some data
        issues = {
            1: IssueInfo(number=1, title="Test", priority="P0", status="completed"),
            2: IssueInfo(number=2, title="Test 2", depends_on={1}, status="pending"),
        }
        state = ImplementationState(
            epic_number=100,
            issues=issues,
            completed_issues={1},
            in_progress={},
            paused_issues={},
        )

        # Verify basic state access
        assert state.epic_number == 100
        assert 1 in state.completed_issues
        assert len(state.issues) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
