#!/usr/bin/env python3
"""Orchestrate parallel implementation of GitHub issues using worktrees.

This script parses an epic issue to extract dependencies, spawns parallel
workers in isolated git worktrees, and uses Claude haiku to implement each issue.

Design goals (following plan_issues.py and analyze_issues_claude.py patterns):
- Dependency-aware scheduling (topological order)
- Thread-safe worktree management
- Live status display with StatusTracker
- Resumable state persistence
- Pause on CI failure for manual intervention
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import datetime as dt
import json
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

MAX_PARALLEL_DEFAULT = 4
ISSUE_TIMEOUT = 1800  # 30 minutes per issue
PR_POLL_INTERVAL = 30  # seconds between PR status checks
PR_MAX_WAIT = 3600  # 1 hour max wait for CI
PLAN_COMMENT_HEADER = "## Detailed Implementation Plan"
MAX_RETRIES = 3
STATUS_REFRESH_INTERVAL = 0.1
API_DELAY_DEFAULT = 1.0  # seconds between GitHub API calls to avoid rate limiting

# Rate limit detection
RATE_LIMIT_RE = re.compile(
    r"Limit reached.*resets\s+(?P<time>[0-9:apm]+)\s*\((?P<tz>[^)]+)\)",
    re.IGNORECASE,
)

ALLOWED_TIMEZONES = {
    "America/Los_Angeles",
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Phoenix",
    "UTC",
    "Europe/London",
    "Europe/Paris",
    "Asia/Tokyo",
}

# Time constants
NOON_HOUR = 12
MIDNIGHT_HOUR = 0


# ---------------------------------------------------------------------
# Secure file helpers
# ---------------------------------------------------------------------


def write_secure(path: pathlib.Path, content: str) -> None:
    """Write content to file with secure permissions (owner-only read/write)."""
    path.write_text(content)
    path.chmod(0o600)


# ---------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------


# Global verbose flag (set from Options)
_verbose_mode = False


def set_verbose(verbose: bool) -> None:
    """Set the global verbose mode."""
    global _verbose_mode
    _verbose_mode = verbose


def log(level: str, msg: str) -> None:
    """Log a message with timestamp and level prefix."""
    if level == "DEBUG" and not _verbose_mode:
        return
    ts = time.strftime("%H:%M:%S")
    out = sys.stderr if level in {"WARN", "ERROR"} else sys.stdout
    print(f"[{level}] {ts} {msg}", file=out, flush=True)


# ---------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------


def run(
    cmd: list[str],
    *,
    timeout: int | None = None,
    cwd: pathlib.Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command and capture output."""
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        cwd=cwd,
    )


# ---------------------------------------------------------------------
# Rate-limit handling (from plan_issues.py)
# ---------------------------------------------------------------------


def parse_reset_epoch(time_str: str, tz: str) -> int:
    """Parse a rate limit reset time string and return epoch seconds."""
    if tz not in ALLOWED_TIMEZONES:
        tz = "America/Los_Angeles"

    now_utc = dt.datetime.now(dt.UTC)
    today = now_utc.astimezone(dt.ZoneInfo(tz)).date()

    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)?$", time_str, re.IGNORECASE)
    if not m:
        return int(time.time()) + 3600

    hour, minute, ampm = m.groups()
    hour = int(hour)
    minute = int(minute or 0)

    if ampm:
        ampm = ampm.lower()
        if ampm == "pm" and hour < NOON_HOUR:
            hour += NOON_HOUR
        if ampm == "am" and hour == NOON_HOUR:
            hour = MIDNIGHT_HOUR

    local = dt.datetime.combine(
        today,
        dt.time(hour, minute),
        tzinfo=dt.ZoneInfo(tz),
    )

    if local < now_utc.astimezone(dt.ZoneInfo(tz)):
        local += dt.timedelta(days=1)

    return int(local.timestamp())


def detect_rate_limit(text: str) -> int | None:
    """Detect rate limit message in text and return reset epoch if found."""
    m = RATE_LIMIT_RE.search(text)
    if not m:
        return None
    return parse_reset_epoch(m.group("time"), m.group("tz"))


def wait_until(epoch: int) -> None:
    """Wait until the given epoch time, showing a countdown."""
    interrupted = False

    def handler(_sig: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    old_handler = signal.signal(signal.SIGINT, handler)
    try:
        while True:
            if interrupted:
                print("\n[INFO] Wait interrupted by user")
                raise KeyboardInterrupt
            remaining = epoch - int(time.time())
            if remaining <= 0:
                print()
                return
            h, r = divmod(remaining, 3600)
            m, s = divmod(r, 60)
            print(f"\r[INFO] Rate limit resets in {h:02d}:{m:02d}:{s:02d}", end="", flush=True)
            time.sleep(1)
    finally:
        signal.signal(signal.SIGINT, old_handler)


# ---------------------------------------------------------------------
# Status Tracker (from plan_issues.py)
# ---------------------------------------------------------------------


class StatusTracker:
    """Thread-safe status tracker for parallel processing with live display."""

    MAIN_THREAD = -1

    def __init__(self, max_workers: int) -> None:
        """Initialize the status tracker."""
        self._lock = threading.Lock()
        self._max_workers = max_workers
        self._slots: dict[int, tuple[int, str, str, float]] = {}
        self._stop_event = threading.Event()
        self._display_thread: threading.Thread | None = None
        self._lines_printed = 0
        self._update_event = threading.Event()
        self._completed_count = 0
        self._total_count = 0
        self._is_tty = sys.stdout.isatty()

    def set_total(self, total: int) -> None:
        """Set the total number of items to process."""
        with self._lock:
            self._total_count = total

    def increment_completed(self) -> None:
        """Increment the completed count."""
        with self._lock:
            self._completed_count += 1
            self._update_event.set()

    def update_main(self, stage: str, info: str = "") -> None:
        """Update the main thread status."""
        with self._lock:
            self._slots[self.MAIN_THREAD] = (0, stage, info, time.time())
            self._update_event.set()

    def acquire_slot(self, item_id: int) -> int:
        """Acquire a display slot for an item. Returns slot number."""
        with self._lock:
            for slot in range(self._max_workers):
                if slot not in self._slots:
                    self._slots[slot] = (item_id, "Starting", "", time.time())
                    self._update_event.set()
                    return slot
            slot = item_id % self._max_workers
            self._slots[slot] = (item_id, "Starting", "", time.time())
            self._update_event.set()
            return slot

    def update(self, slot: int, item_id: int, stage: str, info: str = "") -> None:
        """Update the status for a slot."""
        with self._lock:
            self._slots[slot] = (item_id, stage, info, time.time())
            self._update_event.set()

    def release_slot(self, slot: int) -> None:
        """Release a slot when done."""
        with self._lock:
            if slot in self._slots:
                del self._slots[slot]
            self._update_event.set()

    def _format_elapsed(self, start_time: float) -> str:
        """Format elapsed time as MM:SS."""
        elapsed = max(0, int(time.time() - start_time))
        minutes, seconds = divmod(elapsed, 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _render_status(self) -> list[str]:
        """Render current status as list of lines."""
        with self._lock:
            lines = []

            if self.MAIN_THREAD in self._slots:
                _, stage, info, start_time = self._slots[self.MAIN_THREAD]
                elapsed = self._format_elapsed(start_time)
                progress = f" [{self._completed_count}/{self._total_count}]" if self._total_count > 0 else ""
                info_str = f" - {info}" if info else ""
                lines.append(f"  Main:     [{stage:12}]{progress} ({elapsed}){info_str}")
            else:
                lines.append("  Main:     [Idle        ]")

            for slot in range(self._max_workers):
                if slot in self._slots:
                    item_id, stage, info, start_time = self._slots[slot]
                    elapsed = self._format_elapsed(start_time)
                    info_str = f" - {info}" if info else ""
                    lines.append(f"  Thread {slot}: [{stage:12}] #{item_id} ({elapsed}){info_str}")
                else:
                    lines.append(f"  Thread {slot}: [Idle        ]")
            return lines

    def _display_loop(self) -> None:
        """Background thread that refreshes the status display."""
        last_main_stage = ""
        first_render = True

        while not self._stop_event.is_set():
            lines = self._render_status()
            num_lines = len(lines)

            if self._is_tty:
                if not first_render:
                    sys.stdout.write(f"\033[{num_lines}A")
                for line in lines:
                    sys.stdout.write(f"\r{line}\033[K\n")
                sys.stdout.flush()
                first_render = False
                self._lines_printed = num_lines
            else:
                with self._lock:
                    if self.MAIN_THREAD in self._slots:
                        _, stage, info, _ = self._slots[self.MAIN_THREAD]
                        current = f"{stage}:{info}"
                        if current != last_main_stage:
                            progress = f"[{self._completed_count}/{self._total_count}]" if self._total_count > 0 else ""
                            info_str = f" - {info}" if info else ""
                            print(f"  Progress: {stage}{progress}{info_str}")
                            sys.stdout.flush()
                            last_main_stage = current

            self._update_event.wait(STATUS_REFRESH_INTERVAL)
            self._update_event.clear()

    def start_display(self) -> None:
        """Start the background display thread."""
        if self._is_tty:
            sys.stdout.write("\033[?25l")
            sys.stdout.flush()

        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()
        time.sleep(0.05)

    def stop_display(self) -> None:
        """Stop the background display thread."""
        self._stop_event.set()
        self._update_event.set()
        if self._display_thread:
            self._display_thread.join(timeout=1.0)

        if self._is_tty:
            sys.stdout.write("\033[?25h")
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._lines_printed = 0


# ---------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------


@dataclasses.dataclass
class IssueInfo:
    """Information about a single issue from the epic."""

    number: int
    title: str = ""
    depends_on: set[int] = dataclasses.field(default_factory=set)
    priority: str = "P2"  # P0, P1, P2
    status: str = "pending"  # pending, ready, in_progress, completed, paused, blocked_external

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "number": self.number,
            "title": self.title,
            "depends_on": list(self.depends_on),
            "priority": self.priority,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IssueInfo":
        """Create from dict."""
        return cls(
            number=data["number"],
            title=data.get("title", ""),
            depends_on=set(data.get("depends_on", [])),
            priority=data.get("priority", "P2"),
            status=data.get("status", "pending"),
        )


@dataclasses.dataclass
class PausedIssue:
    """Information about a paused issue."""

    worktree: str
    pr: int | None
    reason: str

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {"worktree": self.worktree, "pr": self.pr, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict) -> "PausedIssue":
        """Create from dict."""
        return cls(
            worktree=data.get("worktree", ""),
            pr=data.get("pr"),
            reason=data.get("reason", ""),
        )


@dataclasses.dataclass
class ImplementationState:
    """Persistent state for resumable execution."""

    epic_number: int
    issues: dict[int, IssueInfo] = dataclasses.field(default_factory=dict)
    completed_issues: set[int] = dataclasses.field(default_factory=set)
    paused_issues: dict[int, PausedIssue] = dataclasses.field(default_factory=dict)
    in_progress: dict[int, str] = dataclasses.field(default_factory=dict)  # issue -> worktree
    pr_numbers: dict[int, int] = dataclasses.field(default_factory=dict)  # issue -> PR number
    started_at: str = ""
    last_updated: str = ""

    def save(self, path: pathlib.Path) -> None:
        """Save state to JSON file."""
        data = {
            "epic_number": self.epic_number,
            "issues": {str(k): v.to_dict() for k, v in self.issues.items()},
            "completed_issues": list(self.completed_issues),
            "paused_issues": {str(k): v.to_dict() for k, v in self.paused_issues.items()},
            "in_progress": {str(k): v for k, v in self.in_progress.items()},
            "pr_numbers": {str(k): v for k, v in self.pr_numbers.items()},
            "started_at": self.started_at,
            "last_updated": dt.datetime.now().isoformat(),
        }
        write_secure(path, json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: pathlib.Path) -> "ImplementationState":
        """Load state from JSON file."""
        if not path.exists():
            return cls(epic_number=0)
        try:
            data = json.loads(path.read_text())
            return cls(
                epic_number=data.get("epic_number", 0),
                issues={int(k): IssueInfo.from_dict(v) for k, v in data.get("issues", {}).items()},
                completed_issues=set(data.get("completed_issues", [])),
                paused_issues={int(k): PausedIssue.from_dict(v) for k, v in data.get("paused_issues", {}).items()},
                in_progress={int(k): v for k, v in data.get("in_progress", {}).items()},
                pr_numbers={int(k): v for k, v in data.get("pr_numbers", {}).items()},
                started_at=data.get("started_at", ""),
                last_updated=data.get("last_updated", ""),
            )
        except (json.JSONDecodeError, KeyError) as e:
            log("WARN", f"Failed to load state: {e}")
            return cls(epic_number=0)


@dataclasses.dataclass
class WorkerResult:
    """Result of processing a single issue."""

    issue: int
    status: str  # completed, paused, skipped, error
    pr_number: int | None = None
    error: str | None = None
    duration: float = 0.0


@dataclasses.dataclass(frozen=True)
class Options:
    """Configuration options for the implementer."""

    epic: int
    priority: str | None  # P0, P1, P2, or None for all
    issues: list[int] | None  # Specific issues to process
    parallel: int  # 1 = sequential, N = parallel workers
    dry_run: bool
    analyze: bool  # Just show dependency graph
    resume: bool
    timeout: int
    cleanup: bool
    state_dir: pathlib.Path | None
    api_delay: float  # seconds between GitHub API calls
    verbose: bool  # Enable verbose/debug output


# ---------------------------------------------------------------------
# Dependency Resolver
# ---------------------------------------------------------------------


class DependencyResolver:
    """Resolves issue dependencies and determines execution order."""

    def __init__(self, issues: dict[int, IssueInfo]) -> None:
        """Initialize the resolver."""
        self.issues = issues
        self._completed: set[int] = set()
        self._paused: set[int] = set()
        self._in_progress: set[int] = set()
        self._lock = threading.Lock()

    def initialize_from_state(self, state: ImplementationState) -> None:
        """Initialize resolver state from persistent state."""
        with self._lock:
            self._completed = state.completed_issues.copy()
            self._paused = set(state.paused_issues.keys())
            self._in_progress = set(state.in_progress.keys())

    def get_all_issue_numbers(self) -> set[int]:
        """Return all issue numbers in the graph."""
        return set(self.issues.keys())

    def get_ready_issues(self) -> list[int]:
        """Return issues whose dependencies are all completed.

        Algorithm:
        1. Filter issues not yet completed, paused, or in progress
        2. For each pending issue, check if all depends_on are in completed set
        3. Filter out issues with external dependencies (not in the epic)
        4. Return issues sorted by priority (P0 first) then by number
        """
        all_issues = self.get_all_issue_numbers()
        priority_order = {"P0": 0, "P1": 1, "P2": 2}

        with self._lock:
            ready = []
            for num, info in self.issues.items():
                # Skip completed, paused, or in-progress issues
                if num in self._completed or num in self._paused or num in self._in_progress:
                    continue

                # Check for external dependencies (not in epic)
                external_deps = info.depends_on - all_issues
                if external_deps:
                    info.status = "blocked_external"
                    continue

                # All internal dependencies must be completed
                if info.depends_on.issubset(self._completed):
                    ready.append(num)

            return sorted(ready, key=lambda n: (priority_order.get(self.issues[n].priority, 2), n))

    def get_blocked_by_external(self) -> dict[int, set[int]]:
        """Return issues blocked by external dependencies."""
        all_issues = self.get_all_issue_numbers()
        blocked = {}
        for num, info in self.issues.items():
            external_deps = info.depends_on - all_issues
            if external_deps:
                blocked[num] = external_deps
        return blocked

    def mark_in_progress(self, issue: int) -> None:
        """Mark an issue as in progress."""
        with self._lock:
            self._in_progress.add(issue)
            self.issues[issue].status = "in_progress"

    def mark_completed(self, issue: int) -> None:
        """Mark an issue as completed, potentially unlocking dependents."""
        with self._lock:
            self._completed.add(issue)
            self._in_progress.discard(issue)
            self.issues[issue].status = "completed"

    def mark_paused(self, issue: int) -> None:
        """Mark an issue as paused (CI failed)."""
        with self._lock:
            self._paused.add(issue)
            self._in_progress.discard(issue)
            self.issues[issue].status = "paused"

    def has_cycle(self) -> bool:
        """Detect cycles using DFS with coloring."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self.issues}

        def dfs(node: int) -> bool:
            color[node] = GRAY
            for dep in self.issues[node].depends_on:
                if dep not in color:
                    continue  # External dependency
                if color[dep] == GRAY:
                    return True  # Back edge = cycle
                if color[dep] == WHITE and dfs(dep):
                    return True
            color[node] = BLACK
            return False

        return any(color[n] == WHITE and dfs(n) for n in self.issues)

    def get_topological_order(self) -> list[int]:
        """Return issues in dependency order (no issue before its dependencies)."""
        priority_order = {"P0": 0, "P1": 1, "P2": 2}
        all_issues = self.get_all_issue_numbers()

        # Kahn's algorithm - only count internal dependencies
        in_degree = {n: len(info.depends_on & all_issues) for n, info in self.issues.items()}
        queue = [n for n, d in in_degree.items() if d == 0]
        result = []

        while queue:
            # Pick by priority, then number
            queue.sort(key=lambda n: (priority_order.get(self.issues[n].priority, 2), n))
            node = queue.pop(0)
            result.append(node)

            for other, info in self.issues.items():
                if node in info.depends_on:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)

        return result


# ---------------------------------------------------------------------
# Worktree Manager
# ---------------------------------------------------------------------


class WorktreeManager:
    """Thread-safe worktree creation and cleanup."""

    def __init__(self, repo_root: pathlib.Path) -> None:
        """Initialize the manager."""
        self.repo_root = repo_root
        self.worktree_base = repo_root / "worktrees"
        self._lock = threading.Lock()

        # Ensure worktrees directory exists
        self.worktree_base.mkdir(exist_ok=True)

    def create(self, issue: int, description: str) -> pathlib.Path:
        """Create a worktree for an issue. Thread-safe."""
        with self._lock:
            # Sanitize description for branch name
            safe_desc = re.sub(r"[^a-zA-Z0-9-]", "-", description.lower())[:30]
            branch_name = f"{issue}-{safe_desc}"
            worktree_path = self.worktree_base / branch_name

            if worktree_path.exists():
                log("WARN", f"Worktree already exists: {worktree_path}")
                return worktree_path

            # Check if branch exists
            cp = run(["git", "show-ref", "--verify", f"refs/heads/{branch_name}"], cwd=self.repo_root)

            if cp.returncode == 0:
                # Branch exists, use it
                cp = run(["git", "worktree", "add", str(worktree_path), branch_name], cwd=self.repo_root)
            else:
                # Create new branch from main
                cp = run(
                    ["git", "worktree", "add", "-b", branch_name, str(worktree_path), "origin/main"], cwd=self.repo_root
                )

            if cp.returncode != 0:
                raise RuntimeError(f"Failed to create worktree: {cp.stderr}")

            return worktree_path

    def remove(self, issue: int) -> bool:
        """Remove a worktree by issue number. Thread-safe."""
        with self._lock:
            # Find worktree matching issue
            for path in self.worktree_base.iterdir():
                if path.name.startswith(f"{issue}-"):
                    branch_name = path.name
                    cp = run(["git", "worktree", "remove", "--force", str(path)], cwd=self.repo_root)
                    if cp.returncode != 0:
                        log("WARN", f"Failed to remove worktree: {cp.stderr}")
                        return False

                    # Delete local branch
                    run(["git", "branch", "-D", branch_name], cwd=self.repo_root)

                    return True
            return False

    def create_for_existing_branch(self, issue: int, remote_branch: str) -> pathlib.Path:
        """Create a worktree for an existing remote branch. Thread-safe."""
        with self._lock:
            worktree_path = self.worktree_base / f"{issue}-{remote_branch}"

            if worktree_path.exists():
                log("DEBUG", f"Worktree already exists: {worktree_path}")
                # Make sure it's on the right branch and up to date
                run(["git", "fetch", "origin", remote_branch], cwd=worktree_path)
                run(["git", "reset", "--hard", f"origin/{remote_branch}"], cwd=worktree_path)
                return worktree_path

            # Fetch the remote branch first
            run(["git", "fetch", "origin", remote_branch], cwd=self.repo_root)

            # Create worktree tracking the remote branch
            cp = run(
                [
                    "git",
                    "worktree",
                    "add",
                    "--track",
                    "-b",
                    remote_branch,
                    str(worktree_path),
                    f"origin/{remote_branch}",
                ],
                cwd=self.repo_root,
            )

            if cp.returncode != 0:
                # Branch might already exist locally, try without -b
                cp = run(
                    ["git", "worktree", "add", str(worktree_path), remote_branch],
                    cwd=self.repo_root,
                )
                if cp.returncode != 0:
                    raise RuntimeError(f"Failed to create worktree for {remote_branch}: {cp.stderr}")

            return worktree_path

    def list_active(self) -> dict[int, pathlib.Path]:
        """List active worktrees mapped to issue numbers."""
        result = {}
        cp = run(["git", "worktree", "list", "--porcelain"], cwd=self.repo_root)

        for line in cp.stdout.split("\n"):
            if line.startswith("worktree "):
                path = pathlib.Path(line.split(" ", 1)[1])
                if path.parent == self.worktree_base:
                    # Extract issue number from name
                    match = re.match(r"(\d+)-", path.name)
                    if match:
                        result[int(match.group(1))] = path

        return result

    def cleanup_stale(self) -> int:
        """Remove stale worktrees. Returns count of removed worktrees."""
        count = 0
        cp = run(["git", "worktree", "prune"], cwd=self.repo_root)
        if cp.returncode == 0:
            log("INFO", "Pruned stale worktrees")

        # Remove any leftover directories
        for path in self.worktree_base.iterdir():
            if not path.is_dir():
                continue
            # Check if this worktree is registered
            cp = run(["git", "worktree", "list", "--porcelain"], cwd=self.repo_root)
            if str(path) not in cp.stdout:
                shutil.rmtree(path, ignore_errors=True)
                count += 1
                log("INFO", f"Removed orphaned worktree: {path}")

        return count


# ---------------------------------------------------------------------
# Issue Implementer
# ---------------------------------------------------------------------


class IssueImplementer:
    """Implements individual issues using Claude sub-agents."""

    def __init__(
        self,
        repo_root: pathlib.Path,
        tempdir: pathlib.Path,
        opts: Options,
        state: ImplementationState,
        resolver: DependencyResolver,
        worktree_manager: WorktreeManager,
        status_tracker: StatusTracker | None = None,
    ) -> None:
        """Initialize the implementer."""
        self.repo_root = repo_root
        self.tempdir = tempdir
        self.opts = opts
        self.state = state
        self.resolver = resolver
        self.worktree_manager = worktree_manager
        self.status_tracker = status_tracker
        self.state_file = (opts.state_dir or tempdir) / "implementation_state.json"

        self._api_lock = threading.Lock()
        self._last_api_call = 0.0

    def _gh_call(self, args: list[str], retries: int = MAX_RETRIES) -> subprocess.CompletedProcess:
        """Make a rate-limited GitHub CLI call with retries."""
        with self._api_lock:
            # Wait for rate limit delay
            elapsed = time.time() - self._last_api_call
            if elapsed < self.opts.api_delay:
                time.sleep(self.opts.api_delay - elapsed)
            self._last_api_call = time.time()

        # Execute with retries
        for attempt in range(1, retries + 1):
            cp = run(args)
            if cp.returncode == 0:
                return cp

            # Check for network/rate limit errors
            stderr_lower = cp.stderr.lower()
            is_network_error = any(
                msg in stderr_lower
                for msg in ["connection", "network", "timeout", "eof", "dns", "resolve", "rate limit"]
            )
            if is_network_error and attempt < retries:
                delay = 2**attempt
                log("WARN", f"API call failed (attempt {attempt}/{retries}), retrying in {delay}s...")
                time.sleep(delay)
                continue
            break
        return cp

    def _fetch_issue_title(self, issue_num: int) -> str:
        """Fetch title for a single issue (rate-limited)."""
        cp = self._gh_call(["gh", "issue", "view", str(issue_num), "--json", "title"])
        if cp.returncode == 0:
            return json.loads(cp.stdout).get("title", f"Issue #{issue_num}")
        return f"Issue #{issue_num}"

    def fetch_titles_for_issues(self, issue_nums: list[int]) -> None:
        """Fetch titles for a list of issues (rate-limited, with progress)."""
        if not issue_nums:
            return

        # Filter to only issues that don't have titles yet
        to_fetch = [n for n in issue_nums if n in self.state.issues and not self.state.issues[n].title]
        if not to_fetch:
            return

        log("INFO", f"Fetching titles for {len(to_fetch)} issues...")
        for idx, issue_num in enumerate(to_fetch, 1):
            if idx % 5 == 0 or idx == len(to_fetch):
                print(f"\r  Fetching: {idx}/{len(to_fetch)}", end="", flush=True)
            title = self._fetch_issue_title(issue_num)
            if issue_num in self.state.issues:
                self.state.issues[issue_num].title = title
        print()  # Newline after progress

    def parse_epic(self, epic_number: int) -> dict[int, IssueInfo]:
        """Parse an epic issue to extract issues with dependencies."""
        log("INFO", f"Parsing epic #{epic_number}...")

        # Retry with exponential backoff for network errors
        for attempt in range(1, MAX_RETRIES + 1):
            cp = run(["gh", "issue", "view", str(epic_number), "--json", "body,title"])
            if cp.returncode == 0:
                break

            # Check for network-related errors
            stderr_lower = cp.stderr.lower()
            is_network_error = any(
                msg in stderr_lower for msg in ["connection", "network", "timeout", "eof", "dns", "resolve"]
            )

            if is_network_error:
                if attempt < MAX_RETRIES:
                    delay = 2**attempt
                    log("WARN", f"Network error fetching epic #{epic_number} (attempt {attempt}/{MAX_RETRIES})")
                    log("INFO", f"Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    log("ERROR", "Network connection failed. Please check:")
                    log("ERROR", "  1. Your internet connection")
                    log("ERROR", "  2. GitHub status: https://githubstatus.com")
                    log("ERROR", "  3. Your VPN/proxy settings")
                    sys.exit(1)
            else:
                # Non-network error, fail immediately
                log("ERROR", f"Failed to fetch epic #{epic_number}: {cp.stderr}")
                sys.exit(1)

        data = json.loads(cp.stdout)
        body = data.get("body", "")

        issues: dict[int, IssueInfo] = {}
        current_priority = "P2"

        # Match priority section headers
        priority_re = re.compile(r"###\s*(P[012]):\s*(\w+)")
        # Match issue lines: - [ ] #123 or - [x] #123 (depends on: #456, #789)
        issue_re = re.compile(r"-\s*\[([x ])\]\s*#(\d+)(?:\s*\(depends on:\s*([^)]+)\))?")

        for line in body.split("\n"):
            # Check for priority header
            priority_match = priority_re.search(line)
            if priority_match:
                current_priority = priority_match.group(1)
                continue

            # Check for issue line
            issue_match = issue_re.search(line)
            if issue_match:
                checked, issue_num_str, deps_str = issue_match.groups()
                issue_num = int(issue_num_str)

                # Parse dependencies
                depends_on: set[int] = set()
                if deps_str:
                    for dep_match in re.finditer(r"#(\d+)", deps_str):
                        depends_on.add(int(dep_match.group(1)))

                # Mark as completed if checked
                status = "completed" if checked == "x" else "pending"

                issues[issue_num] = IssueInfo(
                    number=issue_num,
                    depends_on=depends_on,
                    priority=current_priority,
                    status=status,
                )

        # Titles are fetched lazily when needed (to avoid rate limiting)
        log("INFO", f"Parsed {len(issues)} issues from epic #{epic_number}")
        return issues

    def _update_status(self, slot: int, issue: int, stage: str, info: str = "") -> None:
        """Update status tracker if available."""
        if self.status_tracker and slot >= 0:
            self.status_tracker.update(slot, issue, stage, info)

    def _check_or_create_plan(self, issue: int, slot: int) -> str | None:
        """Check if issue has a plan, create one if missing. Returns plan content."""
        self._update_status(slot, issue, "Plan", "checking")

        # Fetch issue comments
        cp = run(["gh", "issue", "view", str(issue), "--json", "comments"])
        if cp.returncode != 0:
            return None

        data = json.loads(cp.stdout)
        comments = data.get("comments", [])

        # Look for plan comment
        for comment in comments:
            body = comment.get("body", "")
            if PLAN_COMMENT_HEADER in body:
                log("INFO", f"  Found existing plan for #{issue}")
                return body

        # No plan found, generate one
        log("INFO", f"  No plan found for #{issue}, generating...")
        self._update_status(slot, issue, "Plan", "generating")

        # Call plan_issues.py to generate plan
        scripts_dir = self.repo_root / "scripts"
        cp = run(
            ["python3", str(scripts_dir / "plan_issues.py"), "--issues", str(issue), "--auto"],
            timeout=600,  # 10 minute timeout for plan generation
            cwd=self.repo_root,
        )

        if cp.returncode != 0:
            log("WARN", f"  Failed to generate plan for #{issue}: {cp.stderr}")
            return None

        # Fetch the newly created plan
        cp = run(["gh", "issue", "view", str(issue), "--json", "comments"])
        if cp.returncode != 0:
            return None

        data = json.loads(cp.stdout)
        for comment in data.get("comments", []):
            body = comment.get("body", "")
            if PLAN_COMMENT_HEADER in body:
                log("INFO", f"  Generated plan for #{issue}")
                return body

        return None

    def _spawn_claude_agent(
        self,
        worktree: pathlib.Path,
        prompt: str,
        slot: int,
        issue: int,
    ) -> tuple[bool, str]:
        """Spawn a Claude haiku agent to implement the issue."""
        self._update_status(slot, issue, "Claude", "starting")

        cmd = [
            "claude",
            "--model",
            "haiku",
            "--permission-mode",
            "bypassPermissions",
            "--allowedTools",
            "Read,Write,Edit,Glob,Grep,Bash",
            "--add-dir",
            str(worktree),
            "-p",
            prompt,
        ]

        log_file = self.tempdir / f"issue-{issue}-claude.log"
        log("DEBUG", f"  Claude command: {' '.join(cmd[:6])}...")
        log("DEBUG", f"  Claude log file: {log_file}")

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=worktree,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            output_lines = []
            deadline = time.time() + self.opts.timeout
            line_count = 0

            with log_file.open("w") as log_handle:
                for line in proc.stdout:
                    if time.time() > deadline:
                        proc.kill()
                        log("DEBUG", "  Claude timed out")
                        return False, "Timeout exceeded"

                    log_handle.write(line)
                    output_lines.append(line)
                    line_count += 1

                    # In verbose mode, print Claude output (but not too much)
                    if self.opts.verbose:
                        # Print significant lines
                        stripped = line.strip()
                        if stripped and not stripped.startswith("{") and len(stripped) < 200:
                            print(f"    [Claude] {stripped[:100]}", flush=True)

                    # Update status based on output patterns
                    if "Editing" in line or "Writing" in line:
                        self._update_status(slot, issue, "Claude", "editing")
                    elif "Running" in line or "Bash" in line:
                        self._update_status(slot, issue, "Claude", "running cmd")

            proc.wait()
            log("DEBUG", f"  Claude finished with {line_count} lines, exit code {proc.returncode}")
            combined = "".join(output_lines)

            # Check for rate limit
            reset = detect_rate_limit(combined)
            if reset:
                wait_until(reset)
                return False, "Rate limited - retry needed"

            if proc.returncode == 0:
                return True, combined
            else:
                return False, f"Exit code {proc.returncode}: {combined[-500:]}"

        except Exception as e:
            return False, str(e)

    def _get_summary(self, worktree: pathlib.Path, slot: int, issue: int) -> str:
        """Get a concise summary of changes from Claude."""
        self._update_status(slot, issue, "Summary", "generating")

        prompt = """Concisely summarize the changes you just made in 2-3 sentences.
Focus on what functionality was added or fixed. Do not include implementation details."""

        cmd = [
            "claude",
            "--model",
            "haiku",
            "-p",
            prompt,
            "--add-dir",
            str(worktree),
        ]

        try:
            cp = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=worktree,
            )
            if cp.returncode == 0:
                return cp.stdout.strip()
        except Exception as e:
            log("WARN", f"  Failed to get summary: {e}")

        return "Implementation completed."

    def _check_existing_pr(self, issue: int) -> dict | None:
        """Check if a PR already exists for this issue.

        Returns PR info dict if found, None otherwise.
        Checks both by issue number in title/body AND by branch name pattern.
        """
        # First try searching by issue number in title/body
        cp = self._gh_call(
            [
                "gh",
                "pr",
                "list",
                "--search",
                f"#{issue} in:title,body",
                "--json",
                "number,state,headRefName,statusCheckRollup",
            ]
        )
        if cp.returncode == 0:
            try:
                prs = json.loads(cp.stdout)
                # Find open PR for this issue
                for pr in prs:
                    if pr.get("state") == "OPEN":
                        return pr
                # Check for merged PRs too
                for pr in prs:
                    if pr.get("state") == "MERGED":
                        return pr
            except json.JSONDecodeError:
                pass

        # Also check by branch name pattern (issue number prefix)
        cp = self._gh_call(
            ["gh", "pr", "list", "--state", "all", "--json", "number,state,headRefName,statusCheckRollup"]
        )
        if cp.returncode == 0:
            try:
                prs = json.loads(cp.stdout)
                for pr in prs:
                    branch = pr.get("headRefName", "")
                    # Check if branch starts with issue number
                    if branch.startswith(f"{issue}-") or branch.startswith(f"{issue}_"):
                        state = pr.get("state", "")
                        if state == "OPEN":
                            log("DEBUG", f"  Found existing PR by branch: #{pr['number']} ({branch})")
                            return pr
                        elif state == "MERGED":
                            log("DEBUG", f"  Found merged PR by branch: #{pr['number']} ({branch})")
                            return pr
            except json.JSONDecodeError:
                pass

        return None

    def _analyze_and_fix_pr(self, pr_info: dict, issue: int, slot: int) -> WorkerResult:
        """Analyze CI failures on an existing PR and attempt to fix them."""
        start_time = time.time()
        pr_number = pr_info["number"]
        branch = pr_info.get("headRefName", "")

        self._update_status(slot, issue, "PR", f"analyzing #{pr_number}")
        log("INFO", f"  Found existing PR #{pr_number} for issue #{issue}")
        log("DEBUG", f"  PR branch: {branch}")

        # Check CI status
        checks = pr_info.get("statusCheckRollup", []) or []
        failing = [c for c in checks if c.get("conclusion") == "FAILURE"]
        log("DEBUG", f"  Total checks: {len(checks)}, failing: {len(failing)}")

        if not failing:
            # No failures - PR is in good state, move to next issue
            log("DEBUG", "  No CI failures detected, PR is in good state")
            log("INFO", f"  PR #{pr_number} has no CI failures - moving to next issue")
            duration = time.time() - start_time
            return WorkerResult(issue, "completed", pr_number, None, duration)

        # Get failure details
        self._update_status(slot, issue, "CI", "fetching logs")
        failure_info = []
        for check in failing:
            name = check.get("name", "unknown")
            failure_info.append(f"- {name}: FAILURE")
            log("DEBUG", f"  Failing check: {name}")

        log("INFO", f"  CI failures: {len(failing)} checks failed")

        # Get or create worktree for the existing PR branch
        log("DEBUG", f"  Setting up worktree for existing branch: {branch}")
        self._update_status(slot, issue, "Worktree", "setting up")
        worktree = self.worktree_manager.create_for_existing_branch(issue, branch)
        log("DEBUG", f"  Worktree ready: {worktree}")
        self.state.in_progress[issue] = str(worktree)
        self.state.pr_numbers[issue] = pr_number
        self.state.save(self.state_file)

        # Fetch CI logs
        log("DEBUG", "  Fetching CI run info")
        self._update_status(slot, issue, "CI", "analyzing failures")
        cp = run(["gh", "run", "list", "--branch", branch, "--limit", "1", "--json", "databaseId"])
        run_id = None
        if cp.returncode == 0:
            try:
                runs = json.loads(cp.stdout)
                if runs:
                    run_id = runs[0].get("databaseId")
                    log("DEBUG", f"  Found CI run ID: {run_id}")
            except json.JSONDecodeError:
                log("DEBUG", "  Failed to parse CI run list")

        log_content = ""
        if run_id:
            log("DEBUG", f"  Fetching failed logs for run {run_id}")
            cp = run(["gh", "run", "view", str(run_id), "--log-failed"], timeout=60)
            if cp.returncode == 0:
                log_content = cp.stdout[-5000:]  # Last 5000 chars
                log("DEBUG", f"  Got {len(log_content)} chars of logs")
            else:
                log("DEBUG", f"  Failed to get logs: {cp.stderr}")
        else:
            log("DEBUG", "  No CI run ID found")

        # Build fix prompt
        log("DEBUG", "  Building fix prompt for Claude")
        fix_prompt = f"""You are fixing CI failures for GitHub issue #{issue}.

PR #{pr_number} has failing CI checks:
{chr(10).join(failure_info)}

Here are the relevant failure logs:
```
{log_content}
```

Please:
1. Analyze the failure logs to understand what's broken
2. Fix the issues in the code
3. Run tests locally if possible to verify
4. Ensure your fixes follow the existing code patterns

When done, all files should be saved.
"""

        # Run Claude to fix
        log("DEBUG", "  Spawning Claude agent to fix issues")
        self._update_status(slot, issue, "Fix", "running Claude")
        success, output = self._spawn_claude_agent(worktree, fix_prompt, slot, issue)
        log("DEBUG", f"  Claude returned: success={success}, output_len={len(output)}")

        if not success:
            log("DEBUG", f"  Claude failed: {output[-200:]}")
            self._post_issue_update(issue, f"⚠️ Failed to fix CI failures automatically.\n\nError: {output[-200:]}")
            duration = time.time() - start_time
            return WorkerResult(issue, "paused", pr_number, "fix failed", duration)

        # Check for changes
        log("DEBUG", "  Checking for changes")
        cp = run(["git", "status", "--porcelain"], cwd=worktree)
        if not cp.stdout.strip():
            log("DEBUG", "  No changes made by Claude")
            self._post_issue_update(issue, "ℹ️ No changes needed - CI may need manual investigation.")
            duration = time.time() - start_time
            return WorkerResult(issue, "paused", pr_number, "no changes made", duration)

        # Commit and push
        log("DEBUG", "  Committing and pushing changes")
        self._update_status(slot, issue, "Git", "pushing fix")
        run(["git", "add", "-A"], cwd=worktree)
        run(["git", "commit", "-m", "fix: Address CI failures\n\nAutomated fix by implement_issues.py"], cwd=worktree)
        cp = run(["git", "push", "origin", branch], cwd=worktree)

        if cp.returncode != 0:
            log("DEBUG", f"  Push failed: {cp.stderr}")
            self._post_issue_update(issue, f"⚠️ Failed to push fix: {cp.stderr}")
            duration = time.time() - start_time
            return WorkerResult(issue, "paused", pr_number, "push failed", duration)

        log("DEBUG", "  Push successful, moving to next issue")
        self._post_issue_update(issue, f"🔧 Pushed fix to PR #{pr_number}. Moving to next issue.")

        # Cleanup worktree since we're moving on
        self.worktree_manager.remove(issue)
        del self.state.in_progress[issue]
        self.state.save(self.state_file)

        duration = time.time() - start_time
        log("INFO", f"  Fixed PR #{pr_number} - moving to next issue")
        return WorkerResult(issue, "completed", pr_number, None, duration)

    def _post_issue_update(self, issue: int, message: str) -> None:
        """Post an update comment to the GitHub issue."""
        comment = f"{message}\n\n---\n*Automated by implement_issues.py at {time.strftime('%Y-%m-%d %H:%M:%S')}*"
        proc = subprocess.Popen(
            ["gh", "issue", "comment", str(issue), "--body-file", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        proc.communicate(input=comment)

    def _poll_pr_status(self, pr_number: int, slot: int, issue: int) -> str:
        """Poll PR status until merged, failed, or timeout.

        Returns: "merged", "failed", "timeout"
        """
        max_wait = PR_MAX_WAIT
        start = time.time()

        while time.time() - start < max_wait:
            self._update_status(slot, issue, "PR", "checking")

            cp = run(["gh", "pr", "view", str(pr_number), "--json", "state,mergeStateStatus,statusCheckRollup"])

            if cp.returncode != 0:
                self._update_status(slot, issue, "PR", "error")
                time.sleep(PR_POLL_INTERVAL)
                continue

            data = json.loads(cp.stdout)
            state = data.get("state", "").upper()
            checks = data.get("statusCheckRollup", []) or []

            if state == "MERGED":
                return "merged"

            if state == "CLOSED":
                return "failed"

            # Count check statuses
            pending = sum(1 for c in checks if c.get("status") == "PENDING" or c.get("status") == "QUEUED")
            failing = sum(1 for c in checks if c.get("conclusion") == "FAILURE")

            if failing > 0:
                self._update_status(slot, issue, "PR", f"CI failing ({failing})")
                return "failed"
            elif pending > 0:
                self._update_status(slot, issue, "PR", f"CI running ({pending})")
            else:
                self._update_status(slot, issue, "PR", "waiting for merge")

            time.sleep(PR_POLL_INTERVAL)

        return "timeout"

    def implement_issue(self, issue: int, slot: int) -> WorkerResult:
        """Full workflow for implementing a single issue."""
        start_time = time.time()
        worktree: pathlib.Path | None = None

        try:
            issue_info = self.state.issues.get(issue)
            if not issue_info:
                return WorkerResult(issue, "error", None, "Issue not found in state", 0)

            # Fetch title if not already fetched (lazy loading)
            if not issue_info.title:
                issue_info.title = self._fetch_issue_title(issue)

            title = issue_info.title or f"Issue #{issue}"
            log("INFO", f"Implementing #{issue}: {title}")

            # 0. Check for existing PR first
            self._update_status(slot, issue, "PR", "checking existing")
            existing_pr = self._check_existing_pr(issue)
            if existing_pr:
                state = existing_pr.get("state", "")
                if state == "MERGED":
                    log("INFO", f"  Issue #{issue} already has merged PR #{existing_pr['number']}")
                    self._post_issue_update(issue, f"✅ Already completed via PR #{existing_pr['number']}")
                    return WorkerResult(issue, "completed", existing_pr["number"], None, time.time() - start_time)
                elif state == "OPEN":
                    # Existing open PR - analyze and fix if needed
                    return self._analyze_and_fix_pr(existing_pr, issue, slot)

            # 1. Check/create plan
            plan = self._check_or_create_plan(issue, slot)
            if not plan:
                log("WARN", f"  Could not get/create plan for #{issue}, skipping")
                return WorkerResult(issue, "skipped", None, "No plan available", time.time() - start_time)

            # 2. Create worktree
            self._update_status(slot, issue, "Worktree", "creating")
            worktree = self.worktree_manager.create(issue, title)
            self.state.in_progress[issue] = str(worktree)
            self.state.save(self.state_file)

            # 3. Fetch and rebase
            self._update_status(slot, issue, "Git", "fetch & rebase")
            cp = run(["git", "fetch", "origin"], cwd=worktree)
            if cp.returncode != 0:
                raise RuntimeError(f"git fetch failed: {cp.stderr}")

            cp = run(["git", "rebase", "origin/main"], cwd=worktree)
            if cp.returncode != 0:
                # Try merge instead
                run(["git", "rebase", "--abort"], cwd=worktree)
                cp = run(["git", "merge", "origin/main", "-m", "Merge origin/main"], cwd=worktree)
                if cp.returncode != 0:
                    raise RuntimeError(f"git rebase/merge failed: {cp.stderr}")

            # 4. Run implementation with retries
            implementation_prompt = f"""You are implementing GitHub issue #{issue}: {title}

Here is the implementation plan:

{plan}

Please implement this plan step by step. Make sure to:
1. Follow the plan exactly
2. Write clean, well-documented code
3. Follow existing code patterns in the repository
4. Run tests if applicable

When you're done, ensure all files are saved.
"""

            success = False
            output = ""
            for attempt in range(1, MAX_RETRIES + 1):
                self._update_status(slot, issue, "Implement", f"attempt {attempt}")
                success, output = self._spawn_claude_agent(worktree, implementation_prompt, slot, issue)
                if success:
                    break
                if attempt < MAX_RETRIES:
                    self._update_status(slot, issue, "Retry", f"in {2**attempt}s")
                    time.sleep(2**attempt)

            if not success:
                raise RuntimeError(f"Implementation failed after {MAX_RETRIES} attempts: {output[-200:]}")

            # 5. Check for changes
            self._update_status(slot, issue, "Git", "checking changes")
            cp = run(["git", "status", "--porcelain"], cwd=worktree)
            if not cp.stdout.strip():
                log("WARN", f"  No changes made for #{issue}")
                self.worktree_manager.remove(issue)
                del self.state.in_progress[issue]
                self.state.save(self.state_file)
                return WorkerResult(issue, "skipped", None, "No changes made", time.time() - start_time)

            # 6. Get summary
            summary = self._get_summary(worktree, slot, issue)

            # 7. Post summary to issue
            self._update_status(slot, issue, "Summary", "posting")
            summary_comment = f"## Implementation Summary\n\n{summary}\n\n---\n*Automated by implement_issues.py*"
            proc = subprocess.Popen(
                ["gh", "issue", "comment", str(issue), "--body-file", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            proc.communicate(input=summary_comment)

            # 8. Commit
            self._update_status(slot, issue, "Git", "committing")
            run(["git", "add", "-A"], cwd=worktree)
            commit_msg = f"feat: Implement #{issue}\n\nCloses #{issue}\n\n{summary}"
            run(["git", "commit", "-m", commit_msg], cwd=worktree)

            # 9. Push
            self._update_status(slot, issue, "Git", "pushing")
            branch = worktree.name
            cp = run(["git", "push", "-u", "origin", branch], cwd=worktree)
            if cp.returncode != 0:
                raise RuntimeError(f"git push failed: {cp.stderr}")

            # 10. Create PR
            self._update_status(slot, issue, "PR", "creating")
            pr_body = f"Closes #{issue}\n\n## Summary\n\n{summary}\n\n## Plan\n\n{plan[:500]}..."
            cp = run(
                [
                    "gh",
                    "pr",
                    "create",
                    "--title",
                    f"Implement #{issue}: {title[:50]}",
                    "--body",
                    pr_body,
                    "--head",
                    branch,
                ]
            )

            if cp.returncode != 0:
                raise RuntimeError(f"PR creation failed: {cp.stderr}")

            pr_url = cp.stdout.strip()
            pr_number = int(pr_url.split("/")[-1])
            self.state.pr_numbers[issue] = pr_number
            self.state.save(self.state_file)

            # 11. Enable auto-merge
            run(["gh", "pr", "merge", str(pr_number), "--auto", "--rebase"])

            # 12. Cleanup worktree and move to next issue (don't wait for CI)
            self._update_status(slot, issue, "Cleanup", "removing worktree")
            self.worktree_manager.remove(issue)
            del self.state.in_progress[issue]
            self.state.save(self.state_file)

            duration = time.time() - start_time
            log("INFO", f"  Created PR #{pr_number} for #{issue} in {duration:.0f}s - moving to next issue")
            self._post_issue_update(issue, f"✅ Created PR #{pr_number}. Auto-merge enabled.")
            return WorkerResult(issue, "completed", pr_number, None, duration)

        except Exception as e:
            log("ERROR", f"  Issue #{issue} failed: {e}")

            # Post failure to GitHub issue
            error_msg = str(e)[:200]
            self._post_issue_update(issue, f"❌ Implementation failed:\n\n```\n{error_msg}\n```")

            # Cleanup on failure if worktree was created
            if issue in self.state.in_progress:
                # Keep worktree for debugging, just update state
                self.state.paused_issues[issue] = PausedIssue(
                    worktree=self.state.in_progress[issue],
                    pr=self.state.pr_numbers.get(issue),
                    reason=str(e)[:100],
                )
                del self.state.in_progress[issue]
                self.state.save(self.state_file)

            duration = time.time() - start_time
            return WorkerResult(issue, "paused", None, str(e), duration)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def print_analysis(resolver: DependencyResolver, state: ImplementationState) -> None:
    """Print dependency analysis."""
    priority_labels = {"P0": "Foundation", "P1": "Important", "P2": "Nice-to-have"}

    # Group by priority
    by_priority: dict[str, list[IssueInfo]] = {"P0": [], "P1": [], "P2": []}
    for info in state.issues.values():
        by_priority[info.priority].append(info)

    # Sort each group by issue number
    for priority in by_priority:
        by_priority[priority].sort(key=lambda x: x.number)

    print("\nDependency Analysis:")

    blocked_external = resolver.get_blocked_by_external()
    ready = set(resolver.get_ready_issues())

    for priority in ["P0", "P1", "P2"]:
        issues = by_priority[priority]
        if not issues:
            continue

        print(f"\n  {priority} ({priority_labels[priority]}): {len(issues)} issues")

        for info in issues[:10]:  # Show first 10 per priority
            status_str = ""
            if info.number in state.completed_issues:
                status_str = " [DONE]"
            elif info.number in blocked_external:
                ext_deps = blocked_external[info.number]
                status_str = f" - Blocked by: {', '.join(f'#{d}' for d in ext_deps)} (external)"
            elif info.number in ready:
                status_str = " - Ready"
            elif info.depends_on:
                status_str = f" - Depends: {', '.join(f'#{d}' for d in info.depends_on)}"

            print(f"    #{info.number}{status_str}")

        if len(issues) > 10:
            print(f"    ... ({len(issues) - 10} more)")

    completed_count = len(state.completed_issues)
    ready_count = len(ready)
    blocked_count = len(blocked_external)
    paused_count = len(state.paused_issues)

    print("\nSummary:")
    print(f"  Completed: {completed_count}")
    print(f"  Ready to start: {ready_count}")
    print(f"  Blocked by external deps: {blocked_count}")
    print(f"  Paused (CI failed): {paused_count}")


def print_paused_summary(state: ImplementationState) -> None:
    """Print summary of paused issues."""
    if not state.paused_issues:
        return

    print("\n==========================================")
    print("  Paused Issues (require manual intervention)")
    print("==========================================")

    for issue_num, paused in sorted(state.paused_issues.items()):
        info = state.issues.get(issue_num)
        title = info.title if info else f"Issue #{issue_num}"
        print(f"\n  #{issue_num} - {title}")
        if paused.pr:
            print(f"    PR: #{paused.pr}")
        print(f"    Worktree: {paused.worktree}")
        print(f"    Reason: {paused.reason}")

    issue_list = ",".join(str(n) for n in state.paused_issues.keys())
    print("\nTo retry paused issues after fixing:")
    print(f"  python scripts/implement_issues.py --epic {state.epic_number} --resume --issues {issue_list}")


def main() -> int:
    """CLI entry point for the issue implementer."""
    p = argparse.ArgumentParser(
        description="Orchestrate parallel implementation of GitHub issues using worktrees",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --epic 2784 --analyze              Show dependency graph
  %(prog)s --epic 2784 --dry-run              Preview what would be done
  %(prog)s --epic 2784 --priority P0          Run P0 issues only
  %(prog)s --epic 2784 --parallel             Run with 4 parallel workers
  %(prog)s --epic 2784 --parallel 2           Run with 2 parallel workers
  %(prog)s --epic 2784 --resume               Resume from previous run
  %(prog)s --epic 2784 --issues 2719,2720     Run specific issues only
  %(prog)s --epic 2784 --cleanup              Cleanup stale worktrees
""",
    )
    p.add_argument("--epic", type=int, required=True, help="Epic issue number to parse")
    p.add_argument("--priority", choices=["P0", "P1", "P2"], help="Only process issues with this priority")
    p.add_argument("--issues", metavar="N,M,...", help="Only process specific issue numbers")
    p.add_argument(
        "--parallel",
        type=int,
        nargs="?",
        const=MAX_PARALLEL_DEFAULT,
        default=None,
        metavar="N",
        help=f"Run in parallel (default: {MAX_PARALLEL_DEFAULT} workers if N not specified)",
    )
    p.add_argument("--dry-run", action="store_true", help="Preview actions without making changes")
    p.add_argument("--analyze", action="store_true", help="Just show dependency graph")
    p.add_argument("--resume", action="store_true", help="Resume from previous run")
    p.add_argument(
        "--timeout",
        type=int,
        default=ISSUE_TIMEOUT,
        metavar="SEC",
        help=f"Timeout per issue (default: {ISSUE_TIMEOUT})",
    )
    p.add_argument("--cleanup", action="store_true", help="Cleanup stale worktrees and exit")
    p.add_argument("--state-dir", type=pathlib.Path, help="Directory to persist state")
    p.add_argument(
        "--api-delay",
        type=float,
        default=API_DELAY_DEFAULT,
        metavar="SEC",
        help=f"Delay between GitHub API calls (default: {API_DELAY_DEFAULT}s)",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose/debug output",
    )

    args = p.parse_args()

    # Set verbose mode
    set_verbose(args.verbose)

    # Parse issue numbers if provided
    issues: list[int] | None = None
    if args.issues:
        issues = [int(n.strip()) for n in args.issues.split(",")]

    # If --parallel not specified, run sequentially (1 worker)
    # If --parallel specified without value, use default (4)
    # If --parallel N specified, use N
    parallel_workers = args.parallel if args.parallel is not None else 1

    opts = Options(
        epic=args.epic,
        priority=args.priority,
        issues=issues,
        parallel=parallel_workers,
        dry_run=args.dry_run,
        analyze=args.analyze,
        resume=args.resume,
        timeout=args.timeout,
        cleanup=args.cleanup,
        state_dir=args.state_dir,
        api_delay=args.api_delay,
        verbose=args.verbose,
    )

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    tempdir = pathlib.Path(tempfile.mkdtemp(prefix="implement-issues-"))
    tempdir.chmod(0o700)

    # Cleanup handler
    def cleanup_handler() -> None:
        if tempdir.exists() and not opts.state_dir:
            shutil.rmtree(tempdir, ignore_errors=True)

    atexit.register(cleanup_handler)

    def signal_handler(signum: int, _frame: object) -> None:
        cleanup_handler()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize managers
    worktree_manager = WorktreeManager(repo_root)

    # Handle cleanup mode
    if opts.cleanup:
        log("INFO", "Cleaning up stale worktrees...")
        count = worktree_manager.cleanup_stale()
        log("INFO", f"Removed {count} stale worktrees")
        return 0

    # Load or create state
    state_dir = opts.state_dir or tempdir
    state_file = state_dir / "implementation_state.json"

    if opts.resume and state_file.exists():
        log("INFO", f"Resuming from state file: {state_file}")
        state = ImplementationState.load(state_file)
        if state.epic_number != opts.epic:
            log("WARN", f"State file is for epic #{state.epic_number}, not #{opts.epic}")
            state = ImplementationState(epic_number=opts.epic)
    else:
        state = ImplementationState(epic_number=opts.epic)
        state.started_at = dt.datetime.now().isoformat()

    # Parse epic and build issues dict
    if not state.issues:
        # Create a temporary implementer just for parsing
        temp_implementer = IssueImplementer(repo_root, tempdir, opts, state, None, worktree_manager, None)
        state.issues = temp_implementer.parse_epic(opts.epic)

        # Mark already-completed issues
        for issue_num, info in state.issues.items():
            if info.status == "completed":
                state.completed_issues.add(issue_num)

        state.save(state_file)

    # Filter by priority if specified
    if opts.priority:
        filtered = {k: v for k, v in state.issues.items() if v.priority == opts.priority}
        log("INFO", f"Filtered to {len(filtered)} {opts.priority} issues")
        state.issues = filtered

    # Filter by specific issues if specified
    if opts.issues:
        filtered = {k: v for k, v in state.issues.items() if k in opts.issues}
        log("INFO", f"Filtered to {len(filtered)} specified issues")
        state.issues = filtered

    # Create resolver
    resolver = DependencyResolver(state.issues)
    resolver.initialize_from_state(state)

    # Handle analyze mode
    if opts.analyze:
        print_analysis(resolver, state)
        return 0

    # Calculate actual worker count (don't spawn more threads than issues)
    pending_count = len(state.issues) - len(state.completed_issues) - len(state.paused_issues)
    actual_workers = min(opts.parallel, max(1, pending_count))

    # Create status tracker with correct worker count
    status_tracker = StatusTracker(actual_workers)

    # Create implementer
    implementer = IssueImplementer(repo_root, tempdir, opts, state, resolver, worktree_manager, status_tracker)

    print()
    print("==========================================")
    print("  ML Odyssey Issue Implementer")
    print(f"  Epic: #{opts.epic}")
    print(f"  Total issues: {len(state.issues)}")
    print(f"  Pending: {pending_count}")
    mode = "Sequential" if actual_workers == 1 else f"Parallel ({actual_workers} workers)"
    if opts.dry_run:
        mode += " (DRY RUN)"
    print(f"  Mode: {mode}")
    print("==========================================")
    print()

    if opts.dry_run or opts.analyze:
        print_analysis(resolver, state)
        ready = resolver.get_ready_issues()
        if opts.dry_run:
            # Fetch titles only for issues we'll display (first 10)
            implementer.fetch_titles_for_issues(ready[:10])
            print(f"\n[DRY RUN] Would start with {len(ready)} ready issues:")
            for issue_num in ready[:10]:
                info = state.issues.get(issue_num)
                print(f"  #{issue_num} - {info.title if info else 'Unknown'}")
            if len(ready) > 10:
                print(f"  ... ({len(ready) - 10} more)")
        return 0

    results: list[WorkerResult] = []
    status_tracker.set_total(len(state.issues))
    status_tracker.start_display()

    try:
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures: dict = {}
            active_count = 0
            stall_count = 0  # Track iterations without progress
            last_debug_time = 0.0  # Rate-limit debug output

            while True:
                status_tracker.update_main("Processing", f"{len(results)} done")

                # Get ready issues
                ready = resolver.get_ready_issues()

                # Filter out issues already in progress (in futures)
                ready = [n for n in ready if n not in futures.values()]

                # Spawn new tasks up to max_parallel
                spawned_this_iteration = 0
                available = actual_workers - active_count

                # Rate-limited debug output (once per second)
                now = time.time()
                if now - last_debug_time >= 1.0:
                    log(
                        "DEBUG",
                        f"Loop: ready={len(ready)} futures={len(futures)} active={active_count} avail={available} done={len(results)}",
                    )
                    last_debug_time = now
                for issue_num in ready[:available]:
                    slot = status_tracker.acquire_slot(issue_num)
                    resolver.mark_in_progress(issue_num)
                    future = executor.submit(implementer.implement_issue, issue_num, slot)
                    futures[future] = issue_num
                    active_count += 1
                    spawned_this_iteration += 1
                    status_tracker.update_main("Spawning", f"#{issue_num}")

                # Check termination conditions
                if not futures and not ready:
                    # Nothing running and nothing ready - we're done or blocked
                    log("INFO", "All processable issues completed or blocked")
                    break

                if not futures and ready and spawned_this_iteration == 0:
                    # Ready issues exist but we couldn't spawn any - something's wrong
                    stall_count += 1
                    if stall_count > 5:
                        log("WARN", f"Stalled with {len(ready)} ready issues but cannot spawn. Exiting.")
                        break
                    time.sleep(1)
                    continue
                else:
                    stall_count = 0

                # Wait for at least one completion
                done_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        done_futures.append(future)

                if not done_futures:
                    time.sleep(1.0)  # Poll interval
                    continue

                for future in done_futures:
                    issue_num = futures.pop(future)
                    active_count -= 1

                    try:
                        result = future.result()
                    except Exception as e:
                        log("ERROR", f"Future for #{issue_num} raised exception: {e}")
                        result = WorkerResult(issue_num, "paused", None, str(e), 0)

                    results.append(result)
                    status_tracker.increment_completed()

                    if result.status == "completed":
                        resolver.mark_completed(issue_num)
                        log("DEBUG", f"Main loop: marked #{issue_num} as completed")
                    else:
                        resolver.mark_paused(issue_num)
                        log("DEBUG", f"Main loop: marked #{issue_num} as paused ({result.status})")

                    # Release slot (find slot first, then release without holding lock)
                    slot_to_release = None
                    with status_tracker._lock:
                        for slot in range(actual_workers):
                            if slot in status_tracker._slots:
                                item, _, _, _ = status_tracker._slots[slot]
                                if item == issue_num:
                                    slot_to_release = slot
                                    break
                    if slot_to_release is not None:
                        status_tracker.release_slot(slot_to_release)

    finally:
        status_tracker.stop_display()

    # Save final state
    state.save(state_file)

    # Print summary
    completed = sum(1 for r in results if r.status == "completed")
    paused = sum(1 for r in results if r.status == "paused")
    skipped = sum(1 for r in results if r.status == "skipped")
    errors = sum(1 for r in results if r.status == "error")

    print()
    print("==========================================")
    print("  Summary")
    print(f"  Completed: {completed}")
    print(f"  Paused:    {paused}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    print(f"  Total:     {len(results)}")
    if opts.state_dir:
        print()
        print(f"  State file: {state_file}")
    print("==========================================")

    # Print paused issues summary
    print_paused_summary(state)

    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
