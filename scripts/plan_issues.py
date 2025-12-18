#!/usr/bin/env python3
"""
plan_issues.py

Python replacement for plan-issues.sh.

Design goals:
- Exact behavioral parity with Bash version
- No shell injection surface
- Deterministic, debuggable execution
- Better structure, resumability, throttling, JSON output
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import datetime as dt
import io
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from typing import Callable, Optional

# ---------------------------------------------------------------------
# Constants (match bash defaults)
# ---------------------------------------------------------------------

MAX_ISSUES_FETCH = 500
MAX_RETRIES = 3
CLAUDE_MAX_TOOLS = 50
CLAUDE_MAX_STEPS = 50
MAX_BODY_SIZE = 1_048_576  # 1MB limit for issue body
MAX_TITLE_LENGTH = 500

# Shell metacharacters that are dangerous in replan reasons
DANGEROUS_SHELL_CHARS = set(";|&$`<>")

ALLOWED_EDITORS = {"vim", "vi", "emacs", "nano", "code", "subl", "nvim", "helix", "micro", "edit"}

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

RATE_LIMIT_RE = re.compile(
    r"Limit reached.*resets\s+(?P<time>[0-9:apm]+)\s*\((?P<tz>[^)]+)\)",
    re.IGNORECASE,
)

# Plan header constants
PLAN_HEADER = "## Detailed Implementation Plan"
PLAN_HEADER_REVISED = "## Detailed Implementation Plan (Revised)"

# Status display refresh interval (faster for responsive updates)
STATUS_REFRESH_INTERVAL = 0.1


# ---------------------------------------------------------------------
# Secure file helpers
# ---------------------------------------------------------------------


def write_secure(path: pathlib.Path, content: str) -> None:
    """Write content to file with secure permissions (owner-only read/write)."""
    path.write_text(content)
    path.chmod(0o600)


# ---------------------------------------------------------------------
# Parallel status tracking
# ---------------------------------------------------------------------


class StatusTracker:
    """Thread-safe status tracker for parallel processing with live display."""

    # Slot ID for main thread
    MAIN_THREAD = -1

    def __init__(self, max_workers: int):
        self._lock = threading.Lock()
        self._max_workers = max_workers
        # slot -> (issue_or_count, stage, info, stage_start_time)
        # For main thread (slot -1), issue_or_count is a count or 0
        self._slots: dict[int, tuple[int, str, str, float]] = {}
        self._stop_event = threading.Event()
        self._display_thread: Optional[threading.Thread] = None
        self._lines_printed = 0
        self._update_event = threading.Event()  # Signal immediate refresh
        self._completed_count = 0
        self._total_count = 0
        # Only use ANSI escape codes if stdout is a real terminal
        self._is_tty = sys.stdout.isatty()

    def set_total(self, total: int) -> None:
        """Set the total number of issues to process."""
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

    def acquire_slot(self, issue: int) -> int:
        """Acquire a display slot for an issue. Returns slot number."""
        with self._lock:
            for slot in range(self._max_workers):
                if slot not in self._slots:
                    self._slots[slot] = (issue, "Starting", "", time.time())
                    self._update_event.set()  # Trigger immediate refresh
                    return slot
            # Fallback: use issue number mod max_workers
            slot = issue % self._max_workers
            self._slots[slot] = (issue, "Starting", "", time.time())
            self._update_event.set()
            return slot

    def update(self, slot: int, issue: int, stage: str, info: str = "") -> None:
        """Update the status for a slot. Triggers immediate display refresh."""
        with self._lock:
            self._slots[slot] = (issue, stage, info, time.time())
            self._update_event.set()  # Trigger immediate refresh

    def release_slot(self, slot: int) -> None:
        """Release a slot when done."""
        with self._lock:
            if slot in self._slots:
                del self._slots[slot]
            self._update_event.set()

    def _format_elapsed(self, start_time: float) -> str:
        """Format elapsed time as MM:SS."""
        elapsed = int(time.time() - start_time)
        minutes, seconds = divmod(elapsed, 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _render_status(self) -> list[str]:
        """Render current status as list of lines."""
        with self._lock:
            lines = []

            # Main thread status (always first)
            if self.MAIN_THREAD in self._slots:
                _, stage, info, start_time = self._slots[self.MAIN_THREAD]
                elapsed = self._format_elapsed(start_time)
                progress = f" [{self._completed_count}/{self._total_count}]" if self._total_count > 0 else ""
                info_str = f" - {info}" if info else ""
                lines.append(f"  Main:     [{stage:12}]{progress} ({elapsed}){info_str}")
            else:
                lines.append(f"  Main:     [Idle        ]")

            # Worker thread statuses
            for slot in range(self._max_workers):
                if slot in self._slots:
                    issue, stage, info, start_time = self._slots[slot]
                    elapsed = self._format_elapsed(start_time)
                    info_str = f" - {info}" if info else ""
                    lines.append(f"  Thread {slot}: [{stage:12}] #{issue} ({elapsed}){info_str}")
                else:
                    lines.append(f"  Thread {slot}: [Idle        ]")
            return lines

    def _display_loop(self) -> None:
        """Background thread that refreshes the status display."""
        last_main_stage = ""
        while not self._stop_event.is_set():
            lines = self._render_status()

            if self._is_tty:
                # Move cursor up and clear lines if we've printed before
                if self._lines_printed > 0:
                    sys.stdout.write(f"\033[{self._lines_printed}A")  # Move up
                    sys.stdout.write("\033[J")  # Clear to end of screen

                # Print status lines
                for line in lines:
                    print(line)
                sys.stdout.flush()

                self._lines_printed = len(lines)
            else:
                # In non-TTY mode, print a simple progress line when stage changes
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

            # Wait for either timeout or explicit update signal
            self._update_event.wait(STATUS_REFRESH_INTERVAL)
            self._update_event.clear()

    def start_display(self) -> None:
        """Start the background display thread."""
        # Force an initial render immediately so display is visible right away
        lines = self._render_status()
        for line in lines:
            print(line)
        sys.stdout.flush()
        self._lines_printed = len(lines)

        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()

    def stop_display(self) -> None:
        """Stop the background display thread."""
        self._stop_event.set()
        self._update_event.set()  # Wake up the display thread
        if self._display_thread:
            self._display_thread.join(timeout=1.0)
        # Clear the status display (only in TTY mode)
        if self._is_tty and self._lines_printed > 0:
            sys.stdout.write(f"\033[{self._lines_printed}A")
            sys.stdout.write("\033[J")
            sys.stdout.flush()
            self._lines_printed = 0


# ---------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------


def log(level: str, msg: str) -> None:
    # DEBUG level only prints if DEBUG env var is set
    if level == "DEBUG" and not os.environ.get("DEBUG"):
        return
    ts = time.strftime("%H:%M:%S")
    out = sys.stderr if level in {"WARN", "ERROR"} else sys.stdout
    print(f"[{level}] {ts} {msg}", file=out, flush=True)


# ---------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------


def run(cmd: list[str], *, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


# ---------------------------------------------------------------------
# Rate-limit handling (matches bash logic)
# ---------------------------------------------------------------------


def parse_reset_epoch(time_str: str, tz: str) -> int:
    """Parse a rate limit reset time string and return epoch seconds.

    Args:
        time_str: Time string like "2pm", "2:30pm", or "14:00"
        tz: Timezone string like "America/Los_Angeles"

    Returns:
        Unix timestamp (epoch seconds) when the rate limit resets
    """
    if tz not in ALLOWED_TIMEZONES:
        tz = "America/Los_Angeles"

    now_utc = dt.datetime.now(dt.timezone.utc)
    today = now_utc.astimezone(dt.ZoneInfo(tz)).date()

    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)?$", time_str, re.I)
    if not m:
        return int(time.time()) + 3600

    hour, minute, ampm = m.groups()
    hour = int(hour)
    minute = int(minute or 0)

    if ampm:
        ampm = ampm.lower()
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0

    local = dt.datetime.combine(
        today,
        dt.time(hour, minute),
        tzinfo=dt.ZoneInfo(tz),
    )

    if local < now_utc.astimezone(dt.ZoneInfo(tz)):
        local += dt.timedelta(days=1)

    return int(local.timestamp())


def detect_rate_limit(text: str) -> Optional[int]:
    """Detect rate limit message in text and return reset epoch if found.

    Args:
        text: Text to search for rate limit patterns

    Returns:
        Unix timestamp when rate limit resets, or None if not rate limited
    """
    m = RATE_LIMIT_RE.search(text)
    if not m:
        return None
    return parse_reset_epoch(m.group("time"), m.group("tz"))


def wait_until(epoch: int) -> None:
    """Wait until the given epoch time, showing a countdown. Raises KeyboardInterrupt if interrupted."""
    interrupted = False

    def handler(sig, frame):
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
# GitHub helpers
# ---------------------------------------------------------------------


def gh_issue_json(issue: int) -> dict:
    """Fetch issue data from GitHub as JSON.

    Args:
        issue: GitHub issue number

    Returns:
        Dictionary with title, body, and comments

    Raises:
        RuntimeError: If all retry attempts fail
    """
    for attempt in range(1, MAX_RETRIES + 1):
        cp = run(["gh", "issue", "view", str(issue), "--json", "title,body,comments"])
        if cp.returncode == 0:
            return json.loads(cp.stdout)

        # Retry on transient network errors
        if attempt < MAX_RETRIES:
            delay = 2**attempt
            log("WARN", f"GitHub API error fetching #{issue} (attempt {attempt}/{MAX_RETRIES}): {cp.stderr.strip()}")
            log("INFO", f"Retrying in {delay}s...")
            time.sleep(delay)
        else:
            raise RuntimeError(f"GitHub API failed after {MAX_RETRIES} attempts: {cp.stderr.strip()}")

    # Should not reach here, but satisfy type checker
    raise RuntimeError("Unexpected error in gh_issue_json")


# ---------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Options:
    auto: bool
    replan: bool
    replan_reason: Optional[str]
    dry_run: bool
    cleanup: bool
    parallel: bool
    max_parallel: int
    timeout: int
    throttle_seconds: float
    json_output: bool


@dataclasses.dataclass
class Result:
    issue: int
    status: str
    error: Optional[str] = None


# ---------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------


class Planner:
    def __init__(
        self,
        repo_root: pathlib.Path,
        tempdir: pathlib.Path,
        opts: Options,
        status_tracker: Optional[StatusTracker] = None,
    ):
        self.repo_root = repo_root
        self.tempdir = tempdir
        self.opts = opts
        self.status_tracker = status_tracker

        prompt_path = repo_root / ".claude/agents/chief-architect.md"
        if not prompt_path.exists():
            raise RuntimeError(f"Missing system prompt: {prompt_path}")
        self.system_prompt = prompt_path.read_text()

        self._throttle_lock = threading.Lock()
        self._last_call = 0.0

    # --------------------------------------------------------------

    def create_issue_files(self, issue: int) -> dict[str, pathlib.Path]:
        """Create secure diagnostic files for an issue (matches bash behavior).

        Args:
            issue: The GitHub issue number

        Returns:
            Dictionary with paths to plan, log, cmd, and prompt files
        """
        files = {
            "plan": self.tempdir / f"issue-{issue}-plan.md",
            "log": self.tempdir / f"issue-{issue}-claude.log",
            "cmd": self.tempdir / f"issue-{issue}-command.sh",
            "prompt": self.tempdir / f"issue-{issue}-prompt.txt",
        }
        for path in files.values():
            path.touch(mode=0o600)
        return files

    # --------------------------------------------------------------

    def fetch_issues(self, explicit: Optional[list[int]], limit: Optional[int]) -> list[int]:
        """Fetch list of issue numbers to process.

        Args:
            explicit: If provided, use these specific issue numbers
            limit: Maximum number of issues to return (applied after sorting)

        Returns:
            List of GitHub issue numbers sorted in ascending order

        Raises:
            RuntimeError: If all retry attempts fail
        """
        if explicit:
            return explicit

        for attempt in range(1, MAX_RETRIES + 1):
            cp = run(["gh", "issue", "list", "--state", "open", "--limit", str(MAX_ISSUES_FETCH), "--json", "number"])
            if cp.returncode == 0:
                issues = sorted([i["number"] for i in json.loads(cp.stdout)])
                return issues[:limit] if limit else issues

            # Retry on transient network errors
            if attempt < MAX_RETRIES:
                delay = 2**attempt
                log("WARN", f"GitHub API error (attempt {attempt}/{MAX_RETRIES}): {cp.stderr.strip()}")
                log("INFO", f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"GitHub API failed after {MAX_RETRIES} attempts: {cp.stderr.strip()}")

        # Should not reach here, but satisfy type checker
        raise RuntimeError("Unexpected error in fetch_issues")

    # --------------------------------------------------------------

    def process_issue(self, issue: int, idx: int, total: int, slot: int = -1) -> Result:
        """Process a single GitHub issue: fetch, generate plan, and post.

        Args:
            issue: GitHub issue number
            idx: Current index in processing sequence
            total: Total number of issues being processed
            slot: Status tracker slot (-1 if not using status tracking)

        Returns:
            Result dataclass with issue number, status, and optional error
        """

        def update_status(stage: str, info: str = "") -> None:
            if self.status_tracker and slot >= 0:
                self.status_tracker.update(slot, issue, stage, info)

        try:
            log("INFO", f"[{idx}/{total}] Issue #{issue}")
            update_status("Init", "diag files")

            # Create diagnostic files for this issue
            diag_files = self.create_issue_files(issue)
            log("INFO", f"  Plan file: {diag_files['plan']}")
            log("INFO", f"  Log file:  {diag_files['log']}")

            update_status("Fetching", "GitHub API")
            data = gh_issue_json(issue)

            update_status("Validating", "title/body")
            # Title validation with truncation warning
            title = data.get("title") or "Untitled"
            if len(title) > MAX_TITLE_LENGTH:
                log("WARN", f"Title unusually long ({len(title)} chars), truncating to {MAX_TITLE_LENGTH}")
                title = title[:MAX_TITLE_LENGTH] + "..."

            # Body size validation
            body = data.get("body") or ""
            if len(body) > MAX_BODY_SIZE:
                raise RuntimeError(f"Issue body too large ({len(body)} bytes, max {MAX_BODY_SIZE})")

            # Check for existing plan (with rate-limit detection)
            update_status("Checking", "existing plan")
            if not self.opts.replan:
                for c in data.get("comments", []):
                    comment_body = c.get("body") or ""
                    if "## Detailed Implementation Plan" in comment_body:
                        # Check if existing plan was rate-limited (should retry)
                        if "Limit reached" in comment_body:
                            log("INFO", "Existing plan was rate-limited, will replan...")
                            break  # Continue to generate new plan
                        update_status("Skipped", "has plan")
                        return Result(issue, "skipped")

            # Dry-run: show what would be done without calling Claude
            if self.opts.dry_run:
                update_status("DryRun", "preview")
                log("INFO", f"[DRY RUN] Would generate plan for #{issue}: {title}")
                log("INFO", "Plan preview (first 20 lines):")
                prompt = self.build_prompt(issue, title, body)
                write_secure(diag_files["prompt"], prompt)
                for line in prompt.split("\n")[:20]:
                    print(f"    {line}")
                update_status("Done", "dry-run")
                return Result(issue, "dry-run")

            update_status("Building", "prompt")
            plan = self.generate_plan(issue, title, body, diag_files, update_status)

            if not plan.strip():
                raise RuntimeError("Empty plan")

            if not self.opts.auto:
                update_status("Reviewing")
                plan = self.review_plan(plan)
                if not plan.strip():
                    update_status("Skipped", "empty review")
                    return Result(issue, "skipped")

            update_status("Posting")
            self.post_plan(issue, plan)
            update_status("Done", "posted")
            return Result(issue, "posted")

        except Exception as e:
            update_status("Error", str(e)[:30])
            return Result(issue, "error", str(e))

    # --------------------------------------------------------------

    def throttle(self) -> None:
        """Apply rate limiting between API calls to avoid overwhelming Claude.

        Sleeps if necessary to ensure at least throttle_seconds between calls.
        Thread-safe for use in parallel mode.
        """
        if self.opts.throttle_seconds <= 0:
            return
        with self._throttle_lock:
            now = time.time()
            delta = now - self._last_call
            if delta < self.opts.throttle_seconds:
                time.sleep(self.opts.throttle_seconds - delta)
            self._last_call = time.time()

    # --------------------------------------------------------------

    def generate_plan(
        self,
        issue: int,
        title: str,
        body: str,
        diag_files: dict[str, pathlib.Path],
        update_status: Optional[Callable[[str, str], None]] = None,
    ) -> str:
        """Generate an implementation plan using Claude.

        Args:
            issue: GitHub issue number
            title: Issue title
            body: Issue body text
            diag_files: Dictionary of diagnostic file paths
            update_status: Optional callback to update status display

        Returns:
            Generated plan as a string
        """

        def status(stage: str, info: str = "") -> None:
            if update_status:
                update_status(stage, info)

        status("Generating", "building prompt")
        prompt = self.build_prompt(issue, title, body)

        # Save prompt to diagnostic file
        write_secure(diag_files["prompt"], prompt)

        # Different allowed tools for auto vs interactive mode (match bash behavior)
        if self.opts.auto:
            allowed_tools = "Read,Glob,Grep,WebFetch,WebSearch,Bash"
        else:
            allowed_tools = "Read,Glob,Grep,WebFetch,WebSearch"

        # Save command to diagnostic file for debugging
        cmd_content = f"""# Security: Using subprocess to prevent shell injection
# Prompt file: {diag_files["prompt"]}
claude --model opus \\
       --permission-mode default \\
       --allowedTools "{allowed_tools}" \\
       --add-dir "{self.repo_root}" \\
       --system-prompt "$(cat .claude/agents/chief-architect.md)" \\
       -p "$(cat {diag_files["prompt"]})"
"""
        write_secure(diag_files["cmd"], cmd_content)

        for attempt in range(1, MAX_RETRIES + 1):
            if self.opts.throttle_seconds > 0:
                status("Throttling", f"{self.opts.throttle_seconds}s")
            self.throttle()

            start_time = time.time()
            attempt_info = f"attempt {attempt}/{MAX_RETRIES}" if attempt > 1 else ""
            status("Calling", f"Claude API {attempt_info}".strip())
            log("INFO", f"Generating plan with Claude Opus (timeout: {self.opts.timeout}s)...")

            # Use streaming subprocess with live output (like bash's tee)
            proc = subprocess.Popen(
                [
                    "claude",
                    "--model",
                    "opus",
                    "--permission-mode",
                    "default",
                    "--allowedTools",
                    allowed_tools,
                    "--add-dir",
                    str(self.repo_root),
                    "--system-prompt",
                    self.system_prompt,
                    "-p",
                    prompt,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            output_lines: list[str] = []
            timed_out = False

            status("Streaming", "Claude output")
            print("  -------- Claude Output --------", flush=True)
            try:
                assert proc.stdout is not None
                deadline = time.time() + self.opts.timeout
                log_handle = diag_files["log"].open("a")
                try:
                    for line in proc.stdout:
                        if time.time() > deadline:
                            proc.kill()
                            timed_out = True
                            break
                        print(line, end="", flush=True)
                        log_handle.write(line)
                        output_lines.append(line)
                finally:
                    log_handle.close()
            finally:
                proc.wait()
                print("  --------------------------------", flush=True)

            if timed_out:
                status("Timeout", "retrying...")
                log("WARN", f"Claude timed out after {self.opts.timeout}s, retrying...")
                continue

            elapsed = time.time() - start_time
            combined = "".join(output_lines)

            # Check for Claude CLI JSON error response
            if '"type":"result","subtype":"error"' in combined:
                error_match = re.search(r'"errors":\[([^\]]*)\]', combined)
                error_detail = error_match.group(1) if error_match else "unknown"
                status("CLIError", "retrying...")
                log("WARN", f"Claude CLI error: {error_detail}")
                time.sleep(2**attempt)
                continue

            reset = detect_rate_limit(combined)
            if reset:
                remaining = reset - int(time.time())
                status("RateLimit", f"~{remaining}s wait")
                wait_until(reset)
                continue

            if proc.returncode == 0:
                status("Completed", f"{elapsed:.0f}s")
                log("INFO", f"Generation completed in {elapsed:.0f}s")
                log("INFO", f"Plan size: {len(combined)} bytes")
                # Save plan to diagnostic file
                write_secure(diag_files["plan"], combined)
                return combined

            status("Retrying", f"exit {proc.returncode}")
            log("WARN", f"Claude returned exit code {proc.returncode}, retrying...")
            time.sleep(2**attempt)

        raise RuntimeError("Claude retries exceeded")

    # --------------------------------------------------------------

    def build_prompt(self, issue: int, title: str, body: str) -> str:
        """Build the prompt to send to Claude for plan generation.

        Args:
            issue: GitHub issue number
            title: Issue title
            body: Issue body text

        Returns:
            Formatted prompt string for Claude
        """
        parts = [
            "Create a detailed implementation plan for the following GitHub issue:\n\n",
            f"Issue #{issue}: {title}\n\n",
            body,
        ]

        if self.opts.replan:
            parts.append("\nNOTE: This is a REPLAN request.\n")
            if self.opts.replan_reason:
                parts.append(f"REPLAN REASON: {self.opts.replan_reason}\n")

        parts.append(
            f"""

BUDGET: You have a maximum of {CLAUDE_MAX_TOOLS} tool calls and {CLAUDE_MAX_STEPS} steps.

Output markdown with:
1. Summary
2. Step-by-step implementation tasks
3. Files to modify/create
4. Testing approach
5. Success criteria

End with:
## Resource Usage
"""
        )
        return "".join(parts)

    # --------------------------------------------------------------

    def review_plan(self, plan: str) -> str:
        """Open generated plan in editor for user review.

        Args:
            plan: The generated plan content

        Returns:
            The (possibly modified) plan after editing, or empty string if user deleted content
        """
        editor = os.environ.get("EDITOR", "vim")
        cmd = pathlib.Path(editor).name
        if cmd not in ALLOWED_EDITORS:
            log("WARN", f"Editor '{cmd}' not approved, using vim")
            editor = "vim"

        # Validate editor exists and is executable
        editor_path = shutil.which(editor)
        if not editor_path:
            log("WARN", f"Editor '{editor}' not found, trying vim...")
            editor_path = shutil.which("vim")
            if not editor_path:
                raise RuntimeError("Neither specified editor nor vim found in PATH")

        path = self.tempdir / "review.md"
        write_secure(path, plan)

        subprocess.run([editor_path, str(path)])
        return path.read_text()

    # --------------------------------------------------------------

    def post_plan(self, issue: int, plan: str) -> None:
        """Post the generated plan as a comment on the GitHub issue.

        Args:
            issue: GitHub issue number
            plan: Plan content to post

        Raises:
            RuntimeError: If posting the comment fails
        """
        header = PLAN_HEADER_REVISED if self.opts.replan else PLAN_HEADER
        header += "\n\n"

        if self.opts.replan_reason:
            header += f"**Replan reason:** {self.opts.replan_reason}\n\n"

        proc = subprocess.Popen(
            ["gh", "issue", "comment", str(issue), "--body-file", "-"],
            stdin=subprocess.PIPE,
            text=True,
        )
        assert proc.stdin
        proc.stdin.write(header)
        proc.stdin.write(plan)
        proc.stdin.close()
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError("Failed to post comment")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate and post implementation plans for GitHub issues using Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --limit 5                    First 5 open issues
  %(prog)s --issues 123,456             Specific issues only
  %(prog)s --auto --replan              Auto mode, allow replanning
  %(prog)s --issues 123 --replan-reason 'Need to add error handling'
  %(prog)s --dry-run --issues 123       Preview without calling Claude
  %(prog)s --auto --parallel            Parallel processing (4 jobs)
  %(prog)s --auto --parallel --max-parallel 8   8 concurrent jobs
""",
    )
    p.add_argument("--limit", type=int, metavar="N", help="Only process first N issues")
    p.add_argument("--issues", metavar="N,M,...", help="Only process specific issue numbers")
    p.add_argument("--auto", action="store_true", help="Non-interactive mode: skip editor, auto-post")
    p.add_argument("--replan", action="store_true", help="Re-plan issues with existing plans")
    p.add_argument("--replan-reason", metavar="TXT", help="Re-plan with context (implies --replan)")
    p.add_argument("--dry-run", action="store_true", help="Preview which issues would be processed")
    p.add_argument("--cleanup", action="store_true", help="Delete temp directory on completion")
    p.add_argument("--parallel", action="store_true", help="Process issues in parallel (requires --auto)")
    p.add_argument("--max-parallel", type=int, default=4, metavar="N", help="Max concurrent jobs (default: 4)")
    p.add_argument("--timeout", type=int, default=600, metavar="SEC", help="Timeout per issue (default: 600)")
    p.add_argument("--throttle", type=float, default=0.0, metavar="SEC", help="Seconds between API calls")
    p.add_argument("--json", action="store_true", help="Output results as JSON")

    args = p.parse_args()

    if args.parallel and not args.auto:
        p.error("--parallel requires --auto")

    # Validate issue numbers are numeric
    issues: Optional[list[int]] = None
    if args.issues:
        for i in args.issues.split(","):
            i = i.strip()
            if not i.isdigit():
                p.error(f"Invalid issue number '{i}' - must be numeric")
        issues = [int(i.strip()) for i in args.issues.split(",")]

    # Validate replan reason doesn't contain dangerous shell characters
    if args.replan_reason:
        if any(c in args.replan_reason for c in DANGEROUS_SHELL_CHARS):
            p.error("--replan-reason contains unsafe shell characters")

    opts = Options(
        auto=args.auto,
        replan=args.replan or bool(args.replan_reason),
        replan_reason=args.replan_reason,
        dry_run=args.dry_run,
        cleanup=args.cleanup,
        parallel=args.parallel,
        max_parallel=args.max_parallel,
        timeout=args.timeout,
        throttle_seconds=args.throttle,
        json_output=args.json,
    )

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    tempdir = pathlib.Path(tempfile.mkdtemp(prefix="plan-issues-"))
    tempdir.chmod(0o700)  # Restrict to owner only

    # Register cleanup handler for normal exit and signals (matches bash trap)
    def cleanup_handler() -> None:
        if tempdir.exists():
            if opts.cleanup:
                shutil.rmtree(tempdir, ignore_errors=True)
            else:
                log("INFO", f"Temp directory preserved: {tempdir}")

    atexit.register(cleanup_handler)

    def signal_handler(signum: int, frame: object) -> None:
        cleanup_handler()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Startup status messages (match bash output)
    log("INFO", f"Temp directory: {tempdir}")
    log(
        "INFO",
        f"Mode: {'AUTO (non-interactive, plans auto-posted)' if opts.auto else 'INTERACTIVE (editor review before posting)'}",
    )
    if opts.parallel:
        log("INFO", f"Parallel: ENABLED ({opts.max_parallel} jobs)")
    if opts.dry_run:
        log("INFO", "Dry-run: ENABLED (no changes will be made to GitHub)")
    if opts.cleanup:
        log("INFO", "Cleanup: ENABLED (temp files deleted on success)")
    if opts.replan:
        log("INFO", "Replan: ENABLED (will re-plan issues with existing plans)")
        if opts.replan_reason:
            log("INFO", f"Replan reason: {opts.replan_reason}")
    else:
        log("INFO", "Replan: DISABLED (will skip issues with existing plans)")

    # Create status tracker for parallel mode
    status_tracker: Optional[StatusTracker] = None
    if opts.parallel:
        status_tracker = StatusTracker(opts.max_parallel)

    planner = Planner(repo_root, tempdir, opts, status_tracker)
    issue_list = planner.fetch_issues(issues, args.limit)

    print()
    print("==========================================")
    print("  Issue Planning Script")
    print(f"  Total issues to process: {len(issue_list)}")
    if opts.parallel:
        print(f"  Parallel jobs: {opts.max_parallel}")
    print("==========================================")
    print()

    results: list[Result] = []

    if opts.parallel and status_tracker:
        # Set up tracking for main thread progress
        status_tracker.set_total(len(issue_list))
        status_tracker.update_main("Initializing")

        # Start the live status display
        status_tracker.start_display()

        # Process with status tracking
        def process_with_status(issue: int, idx: int, total: int) -> tuple[Result, str, int]:
            slot = status_tracker.acquire_slot(issue)
            buffer = io.StringIO()
            try:
                with redirect_stdout(buffer), redirect_stderr(buffer):
                    result = planner.process_issue(issue, idx, total, slot)
                return (result, buffer.getvalue(), slot)
            finally:
                status_tracker.release_slot(slot)
                status_tracker.increment_completed()

        try:
            with ThreadPoolExecutor(max_workers=opts.max_parallel) as ex:
                # Spawn all tasks
                status_tracker.update_main("Spawning", f"{len(issue_list)} tasks")
                futures = {}
                for i, issue in enumerate(issue_list):
                    futures[ex.submit(process_with_status, issue, i + 1, len(issue_list))] = i
                    status_tracker.update_main("Spawning", f"#{issue}")

                # Wait for completion
                status_tracker.update_main("Processing")
                indexed_results: dict[int, tuple[Result, str]] = {}
                for f in as_completed(futures):
                    idx = futures[f]
                    result, output, _slot = f.result()
                    indexed_results[idx] = (result, output)

                # Collecting results
                status_tracker.update_main("Collecting", "results")

                # Stop the display before printing results
                status_tracker.stop_display()

                # Print output in issue order
                for i in range(len(issue_list)):
                    result, output = indexed_results[i]
                    print(output, end="")
                    results.append(result)
        except Exception:
            status_tracker.stop_display()
            raise
    else:
        for i, issue in enumerate(issue_list):
            results.append(planner.process_issue(issue, i + 1, len(issue_list)))

    # Summary with proper error/skip distinction
    if opts.json_output:
        print(json.dumps([dataclasses.asdict(r) for r in results], indent=2))
    else:
        posted = sum(1 for r in results if r.status == "posted")
        errored = sum(1 for r in results if r.status == "error")
        skipped = sum(1 for r in results if r.status in ("skipped", "dry-run"))

        print()
        print("==========================================")
        print("  Summary")
        print(f"  Posted:  {posted}")
        print(f"  Skipped: {skipped}")
        print(f"  Errors:  {errored}")
        print(f"  Total:   {len(results)}")
        if not opts.cleanup:
            print()
            print(f"  Temp directory: {tempdir}")
        print("==========================================")

    # Cleanup is handled by atexit handler registered earlier
    # Return non-zero exit code if any errors occurred
    error_count = sum(1 for r in results if r.status == "error")
    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
