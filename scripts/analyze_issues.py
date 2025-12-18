#!/usr/bin/env python3
"""Analyze GitHub issues, build dependency graph, and create prioritized epic.

This script:
1. Fetches all open GitHub issues with plans
2. Extracts dependencies from issue body text
3. Builds a dependency graph and calculates priority order
4. Posts dependency comments to each issue
5. Creates an epic tracking issue with ordered roadmap

Usage:
    python scripts/analyze_issues.py --dry-run          # Preview analysis
    python scripts/analyze_issues.py --post-comments    # Post dependency comments
    python scripts/analyze_issues.py --create-epic      # Create epic tracking issue
    python scripts/analyze_issues.py --post-comments --create-epic  # Full workflow
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from common import Colors

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

MAX_ISSUES_FETCH = 500
MAX_RETRIES = 3
MAX_PARALLEL = 8

# Rate limit detection (reused from plan_issues.py)
RATE_LIMIT_RE = re.compile(
    r"Limit reached.*resets\s+(?P<time>[0-9:apm]+)\s*\((?P<tz>[^)]+)\)",
    re.IGNORECASE,
)

# Dependency patterns in issue body
DEPENDS_ON_RE = re.compile(r"(?:depends?\s+on|requires?)\s+#(\d+)", re.IGNORECASE)
BLOCKS_RE = re.compile(r"blocks?\s+#(\d+)", re.IGNORECASE)
RELATED_RE = re.compile(r"related\s+(?:to\s+)?#(\d+)", re.IGNORECASE)

# Priority patterns
PRIORITY_RE = re.compile(r"\b(P0|P1|P2)\b", re.IGNORECASE)
CRITICAL_RE = re.compile(r"critical\s*path", re.IGNORECASE)

# Category extraction from title
CATEGORY_RE = re.compile(r"^\[([^\]]+)\]")

# Module inference from file paths
MODULE_PATHS = {
    "shared/core/": "core",
    "shared/autograd/": "autograd",
    "shared/training/": "training",
    "shared/data/": "data",
    "examples/": "examples",
    "tests/": "tests",
    "docs/": "docs",
    ".github/": "ci",
    "scripts/": "tooling",
}

# Module priority (lower = higher priority, must implement first)
MODULE_PRIORITY = {
    "core": 0,
    "autograd": 1,
    "training": 2,
    "data": 2,
    "examples": 3,
    "tests": 4,
    "tooling": 4,
    "docs": 5,
    "ci": 5,
    "unknown": 6,
}

# Dependency comment header (used to detect existing comments)
DEPENDENCY_HEADER = "## Dependencies"

# State file for resume capability
STATE_FILE = Path("logs/analyze_issues_state.json")


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------


@dataclasses.dataclass
class Issue:
    """Represents a GitHub issue with extracted metadata."""

    number: int
    title: str
    labels: list[str]
    body: str

    # Extracted fields
    depends_on: set[int] = dataclasses.field(default_factory=set)
    blocks: set[int] = dataclasses.field(default_factory=set)
    related: set[int] = dataclasses.field(default_factory=set)
    priority: int = 9  # 0=P0, 1=P1, 2=P2, 9=unspecified
    category: str = ""
    module: str = "unknown"
    files_affected: list[str] = dataclasses.field(default_factory=list)

    # Computed fields
    is_critical_path: bool = False
    priority_score: float = 0.0
    depth: int = 0


@dataclasses.dataclass
class AnalysisState:
    """Persistent state for resume capability."""

    started_at: str
    issues_fetched: list[int]
    comments_posted: list[int]
    epic_created: Optional[int]

    def save(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "AnalysisState":
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                data = json.load(f)
                return cls(**data)
        return cls(dt.datetime.now().isoformat(), [], [], None)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def log(level: str, msg: str) -> None:
    """Log a message with timestamp."""
    ts = time.strftime("%H:%M:%S")
    color = ""
    if level == "ERROR":
        color = Colors.FAIL
    elif level == "WARN":
        color = Colors.WARNING
    elif level == "INFO":
        color = Colors.OKGREEN
    end = Colors.ENDC if color else ""
    print(f"{color}[{level}] {ts} {msg}{end}", flush=True)


def run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command and capture output."""
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)


def detect_rate_limit(text: str) -> int | None:
    """Detect rate limit and return reset epoch if found."""
    m = RATE_LIMIT_RE.search(text)
    if not m:
        return None
    # Simple fallback: wait 1 hour
    return int(time.time()) + 3600


def wait_until(epoch: int) -> None:
    """Wait until the given epoch time with countdown."""
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
# GitHub API functions
# ---------------------------------------------------------------------


def fetch_all_issues() -> list[dict]:
    """Fetch all open issues from GitHub."""
    log("INFO", f"Fetching up to {MAX_ISSUES_FETCH} open issues...")

    for attempt in range(1, MAX_RETRIES + 1):
        cp = run(
            [
                "gh",
                "issue",
                "list",
                "--state",
                "open",
                "--limit",
                str(MAX_ISSUES_FETCH),
                "--json",
                "number,title,labels,body",
            ]
        )

        if cp.returncode == 0:
            issues = json.loads(cp.stdout)
            log("INFO", f"Fetched {len(issues)} issues")
            return issues

        # Check for rate limit
        reset_epoch = detect_rate_limit(cp.stderr)
        if reset_epoch:
            log("WARN", "Rate limited, waiting...")
            wait_until(reset_epoch)
            continue

        if attempt < MAX_RETRIES:
            delay = 2**attempt
            log("WARN", f"GitHub API error (attempt {attempt}/{MAX_RETRIES}): {cp.stderr.strip()}")
            time.sleep(delay)
        else:
            raise RuntimeError(f"Failed to fetch issues after {MAX_RETRIES} attempts")

    return []


def fetch_issue_comments(issue_number: int) -> list[dict]:
    """Fetch comments for a specific issue."""
    cp = run(["gh", "issue", "view", str(issue_number), "--json", "comments"])
    if cp.returncode == 0:
        data = json.loads(cp.stdout)
        return data.get("comments", [])
    return []


def post_comment(issue_number: int, body: str) -> bool:
    """Post a comment to an issue."""
    cp = run(["gh", "issue", "comment", str(issue_number), "--body", body])
    return cp.returncode == 0


def create_issue(title: str, body: str, labels: list[str]) -> int | None:
    """Create a new issue and return its number."""
    cmd = ["gh", "issue", "create", "--title", title, "--body", body]
    for label in labels:
        cmd.extend(["--label", label])

    cp = run(cmd)
    if cp.returncode == 0:
        # Parse issue number from output (e.g., "https://github.com/owner/repo/issues/123")
        match = re.search(r"/issues/(\d+)", cp.stdout)
        if match:
            return int(match.group(1))
    return None


# ---------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------


def parse_issue(data: dict) -> Issue:
    """Parse raw issue data into Issue object with extracted metadata."""
    issue = Issue(
        number=data["number"],
        title=data["title"],
        labels=[lbl["name"] if isinstance(lbl, dict) else lbl for lbl in data.get("labels", [])],
        body=data.get("body", "") or "",
    )

    body = issue.body

    # Extract dependencies
    issue.depends_on = set(int(m) for m in DEPENDS_ON_RE.findall(body))
    issue.blocks = set(int(m) for m in BLOCKS_RE.findall(body))
    issue.related = set(int(m) for m in RELATED_RE.findall(body))

    # Extract priority
    priority_match = PRIORITY_RE.search(body)
    if priority_match:
        p = priority_match.group(1).upper()
        issue.priority = {"P0": 0, "P1": 1, "P2": 2}.get(p, 9)

    # Check for critical path
    issue.is_critical_path = bool(CRITICAL_RE.search(body))
    if issue.is_critical_path and issue.priority > 0:
        issue.priority = 0  # Critical path issues are P0

    # Extract category from title
    cat_match = CATEGORY_RE.match(issue.title)
    if cat_match:
        issue.category = cat_match.group(1)

    # Infer module from file paths in body
    for path_prefix, module in MODULE_PATHS.items():
        if path_prefix in body:
            issue.module = module
            break

    # Also check labels for module hints
    label_to_module = {
        "core": "core",
        "autograd": "autograd",
        "training": "training",
        "data": "data",
        "examples": "examples",
        "testing": "tests",
        "documentation": "docs",
        "ci-cd": "ci",
    }
    for label in issue.labels:
        if label in label_to_module:
            issue.module = label_to_module[label]
            break

    return issue


# ---------------------------------------------------------------------
# Dependency graph and analysis
# ---------------------------------------------------------------------


class DependencyGraph:
    """Dependency graph for issues."""

    def __init__(self) -> None:
        self.issues: dict[int, Issue] = {}
        self.edges: dict[int, set[int]] = defaultdict(set)  # issue -> dependencies
        self.reverse_edges: dict[int, set[int]] = defaultdict(set)  # issue -> dependents

    def add_issue(self, issue: Issue) -> None:
        """Add an issue to the graph."""
        self.issues[issue.number] = issue

        # Add dependency edges
        for dep in issue.depends_on:
            self.edges[issue.number].add(dep)
            self.reverse_edges[dep].add(issue.number)

        # Add blocking edges (reverse direction)
        for blocked in issue.blocks:
            self.edges[blocked].add(issue.number)
            self.reverse_edges[issue.number].add(blocked)

    def get_dependencies(self, issue_number: int) -> set[int]:
        """Get all issues this issue depends on."""
        return self.edges.get(issue_number, set())

    def get_dependents(self, issue_number: int) -> set[int]:
        """Get all issues that depend on this issue."""
        return self.reverse_edges.get(issue_number, set())

    def calculate_priority_scores(self) -> None:
        """Calculate priority scores for all issues."""
        for issue in self.issues.values():
            score = 0.0

            # Base priority (P0=0, P1=10, P2=20, unset=90)
            score += issue.priority * 10

            # Module priority
            score += MODULE_PRIORITY.get(issue.module, 6) * 3

            # Number of issues blocked (more blocking = lower score = higher priority)
            blocked_count = len(self.get_dependents(issue.number))
            score -= blocked_count * 5

            # Critical path bonus
            if issue.is_critical_path:
                score -= 50

            # Bug fix priority
            if "bug" in issue.labels:
                score -= 10

            issue.priority_score = score

    def topological_sort(self) -> list[int]:
        """Return issues in dependency order (dependencies first)."""
        # Kahn's algorithm with priority-based tie-breaking
        in_degree: dict[int, int] = defaultdict(int)
        for issue_num in self.issues:
            for dep in self.edges[issue_num]:
                if dep in self.issues:  # Only count edges to known issues
                    in_degree[issue_num] += 1

        # Start with issues that have no dependencies
        ready = [num for num in self.issues if in_degree[num] == 0]
        ready.sort(key=lambda n: self.issues[n].priority_score)

        result = []
        while ready:
            # Take the highest priority (lowest score) issue
            current = ready.pop(0)
            result.append(current)

            # Update dependents
            for dependent in self.get_dependents(current):
                if dependent in self.issues:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        ready.append(dependent)
                        ready.sort(key=lambda n: self.issues[n].priority_score)

        # Handle cycles: add remaining issues sorted by priority
        remaining = [n for n in self.issues if n not in result]
        remaining.sort(key=lambda n: self.issues[n].priority_score)
        result.extend(remaining)

        return result

    def detect_cycles(self) -> list[list[int]]:
        """Detect cycles in the dependency graph."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: int, path: list[int]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.edges.get(node, set()):
                if neighbor not in self.issues:
                    continue
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])

            path.pop()
            rec_stack.remove(node)

        for node in self.issues:
            if node not in visited:
                dfs(node, [])

        return cycles


# ---------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------


def generate_dependency_comment(issue: Issue, graph: DependencyGraph) -> str:
    """Generate dependency comment for an issue."""
    lines = [DEPENDENCY_HEADER, ""]

    # Depends on section
    deps = graph.get_dependencies(issue.number)
    valid_deps = [d for d in deps if d in graph.issues]
    if valid_deps:
        lines.append("### Depends On")
        for dep_num in sorted(valid_deps):
            dep_issue = graph.issues[dep_num]
            priority_str = f"P{dep_issue.priority}" if dep_issue.priority < 9 else ""
            if dep_issue.is_critical_path:
                priority_str = "P0 - Critical Path"
            lines.append(
                f"- #{dep_num} - {dep_issue.title} ({priority_str})"
                if priority_str
                else f"- #{dep_num} - {dep_issue.title}"
            )
        lines.append("")

    # Blocks section
    dependents = graph.get_dependents(issue.number)
    valid_dependents = [d for d in dependents if d in graph.issues]
    if valid_dependents:
        lines.append("### Blocks")
        for dep_num in sorted(valid_dependents):
            dep_issue = graph.issues[dep_num]
            lines.append(f"- #{dep_num} - {dep_issue.title}")
        lines.append("")

    # Related section
    if issue.related:
        valid_related = [r for r in issue.related if r in graph.issues]
        if valid_related:
            lines.append("### Related")
            for rel_num in sorted(valid_related):
                if rel_num in graph.issues:
                    rel_issue = graph.issues[rel_num]
                    lines.append(f"- #{rel_num} - {rel_issue.title}")
            lines.append("")

    if len(lines) <= 2:  # Only header
        return ""

    lines.append("---")
    lines.append("*Auto-generated by analyze_issues.py*")

    return "\n".join(lines)


def generate_epic_body(graph: DependencyGraph, ordered_issues: list[int]) -> str:
    """Generate the epic tracking issue body."""
    lines = [
        "# ML Odyssey Implementation Roadmap",
        "",
        "## Overview",
        f"Ordered implementation plan for {len(ordered_issues)} open issues based on dependency analysis.",
        f"**Generated**: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # Find critical path issues
    critical = [n for n in ordered_issues if graph.issues[n].is_critical_path]
    if critical:
        lines.append(f"**Critical Path Issues**: {', '.join(f'#{n}' for n in critical[:5])}")
        lines.append("")

    # Group by module/phase
    phases = {
        "Phase 1: Core Foundations": ["core"],
        "Phase 2: Autograd System": ["autograd"],
        "Phase 3: Training Infrastructure": ["training", "data"],
        "Phase 4: Model Implementations": ["examples"],
        "Phase 5: Testing": ["tests"],
        "Phase 6: Documentation & Tooling": ["docs", "ci", "tooling", "unknown"],
    }

    for phase_name, modules in phases.items():
        phase_issues = [n for n in ordered_issues if graph.issues[n].module in modules]
        if not phase_issues:
            continue

        lines.append(f"## {phase_name} ({len(phase_issues)} issues)")
        lines.append("")

        # Group by priority within phase
        p0_issues = [n for n in phase_issues if graph.issues[n].priority == 0]
        other_issues = [n for n in phase_issues if graph.issues[n].priority != 0]

        if p0_issues:
            lines.append("### Critical Path")
            for num in p0_issues[:10]:  # Limit display
                issue = graph.issues[num]
                lines.append(f"- [ ] #{num} {issue.title}")
            if len(p0_issues) > 10:
                lines.append(f"- ... and {len(p0_issues) - 10} more")
            lines.append("")

        if other_issues:
            lines.append("### Other Issues")
            for num in other_issues[:15]:
                issue = graph.issues[num]
                lines.append(f"- [ ] #{num} {issue.title}")
            if len(other_issues) > 15:
                lines.append(f"- ... and {len(other_issues) - 15} more")
            lines.append("")

    # Dependency summary table
    lines.append("## Dependency Summary")
    lines.append("")
    lines.append("| Issue | Depends On | Blocks | Priority | Module |")
    lines.append("|-------|-----------|--------|----------|--------|")

    for num in ordered_issues[:30]:  # Limit table size
        issue = graph.issues[num]
        deps = graph.get_dependencies(num)
        blocks = graph.get_dependents(num)
        dep_str = ", ".join(f"#{d}" for d in sorted(deps)[:3]) if deps else "-"
        if len(deps) > 3:
            dep_str += f" +{len(deps) - 3}"
        block_str = ", ".join(f"#{b}" for b in sorted(blocks)[:3]) if blocks else "-"
        if len(blocks) > 3:
            block_str += f" +{len(blocks) - 3}"
        priority_str = f"P{issue.priority}" if issue.priority < 9 else "-"
        if issue.is_critical_path:
            priority_str = "**P0**"
        lines.append(f"| #{num} | {dep_str} | {block_str} | {priority_str} | {issue.module} |")

    if len(ordered_issues) > 30:
        lines.append(f"| ... | {len(ordered_issues) - 30} more issues | | | |")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by `python scripts/analyze_issues.py --create-epic`*")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------


def has_dependency_comment(issue_number: int) -> bool:
    """Check if issue already has a dependency comment."""
    comments = fetch_issue_comments(issue_number)
    for comment in comments:
        body = comment.get("body", "")
        if DEPENDENCY_HEADER in body and "Auto-generated by analyze_issues.py" in body:
            return True
    return False


def analyze_issues(
    dry_run: bool = True,
    post_comments: bool = False,
    create_epic: bool = False,
    output_format: str = "text",
    max_parallel: int = MAX_PARALLEL,
    resume: bool = False,
) -> dict:
    """Main analysis workflow."""
    state = AnalysisState.load() if resume else AnalysisState(dt.datetime.now().isoformat(), [], [], None)

    # Fetch issues
    log("INFO", "Starting issue analysis...")
    raw_issues = fetch_all_issues()

    # Parse and build graph
    graph = DependencyGraph()
    for data in raw_issues:
        issue = parse_issue(data)
        graph.add_issue(issue)
        state.issues_fetched.append(issue.number)

    log("INFO", f"Parsed {len(graph.issues)} issues")

    # Calculate priorities
    graph.calculate_priority_scores()

    # Detect cycles
    cycles = graph.detect_cycles()
    if cycles:
        log("WARN", f"Found {len(cycles)} dependency cycles")
        for cycle in cycles[:3]:
            log("WARN", f"  Cycle: {' -> '.join(f'#{n}' for n in cycle)}")

    # Topological sort
    ordered = graph.topological_sort()
    log("INFO", f"Computed priority order for {len(ordered)} issues")

    # Output analysis
    if output_format == "json":
        result = {
            "ordered_issues": ordered,
            "issues": {
                num: {
                    "title": issue.title,
                    "priority": issue.priority,
                    "priority_score": issue.priority_score,
                    "module": issue.module,
                    "depends_on": list(issue.depends_on),
                    "blocks": list(graph.get_dependents(num)),
                    "is_critical_path": issue.is_critical_path,
                }
                for num, issue in graph.issues.items()
            },
            "cycles": cycles,
        }
        print(json.dumps(result, indent=2))
        return result

    elif output_format == "markdown":
        print(generate_epic_body(graph, ordered))
        return {"ordered_issues": ordered}

    else:  # text
        print(f"\n{Colors.BOLD}Issue Priority Order:{Colors.ENDC}")
        print("=" * 60)
        for i, num in enumerate(ordered[:20], 1):
            issue = graph.issues[num]
            deps = graph.get_dependencies(num)
            blocks = graph.get_dependents(num)
            priority = f"P{issue.priority}" if issue.priority < 9 else "  "
            critical = " [CRITICAL]" if issue.is_critical_path else ""
            print(f"{i:3}. #{num:4} [{priority}] {issue.module:8} {issue.title[:40]:<40}{critical}")
            if deps:
                print(f"       Depends on: {', '.join(f'#{d}' for d in sorted(deps)[:5])}")
            if blocks:
                print(f"       Blocks: {', '.join(f'#{b}' for b in sorted(blocks)[:5])}")
        if len(ordered) > 20:
            print(f"... and {len(ordered) - 20} more issues")

    # Post dependency comments
    if post_comments and not dry_run:
        log("INFO", "Posting dependency comments...")
        posted = 0
        skipped = 0

        def post_for_issue(num: int) -> tuple[int, bool, str]:
            if num in state.comments_posted:
                return (num, False, "already posted")
            if has_dependency_comment(num):
                return (num, False, "exists")
            comment = generate_dependency_comment(graph.issues[num], graph)
            if not comment:
                return (num, False, "no deps")
            success = post_comment(num, comment)
            return (num, success, "posted" if success else "failed")

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {executor.submit(post_for_issue, num): num for num in ordered}
            for future in as_completed(futures):
                num, success, status = future.result()
                if success:
                    state.comments_posted.append(num)
                    posted += 1
                    log("INFO", f"Posted comment to #{num}")
                else:
                    skipped += 1
                    if status == "failed":
                        log("WARN", f"Failed to post to #{num}")

        state.save()
        log("INFO", f"Posted {posted} comments, skipped {skipped}")

    elif post_comments and dry_run:
        log("INFO", "[DRY RUN] Would post dependency comments to issues with dependencies")
        count = sum(1 for num in ordered if generate_dependency_comment(graph.issues[num], graph))
        log("INFO", f"[DRY RUN] {count} issues would receive comments")

    # Create epic
    if create_epic and not dry_run:
        log("INFO", "Creating epic tracking issue...")
        epic_body = generate_epic_body(graph, ordered)
        epic_number = create_issue(
            title="[Epic] ML Odyssey Implementation Roadmap",
            body=epic_body,
            labels=["epic", "planning", "tracking"],
        )
        if epic_number:
            state.epic_created = epic_number
            state.save()
            log("INFO", f"Created epic issue #{epic_number}")
        else:
            log("ERROR", "Failed to create epic issue")

    elif create_epic and dry_run:
        log("INFO", "[DRY RUN] Would create epic tracking issue")
        print("\n" + "=" * 60)
        print("EPIC PREVIEW:")
        print("=" * 60)
        print(generate_epic_body(graph, ordered)[:2000])
        print("...")

    return {"ordered_issues": ordered, "graph": graph}


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze GitHub issues, build dependency graph, create epic")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview without making changes (default: True)",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually make changes",
    )
    parser.add_argument(
        "--post-comments",
        action="store_true",
        help="Post dependency comments to issues",
    )
    parser.add_argument(
        "--create-epic",
        action="store_true",
        help="Create epic tracking issue",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=MAX_PARALLEL,
        help=f"Max concurrent API calls (default: {MAX_PARALLEL})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved state",
    )

    args = parser.parse_args()

    # Handle dry-run logic
    dry_run = args.dry_run and not args.no_dry_run

    if not sys.stdout.isatty():
        Colors.disable()

    try:
        analyze_issues(
            dry_run=dry_run,
            post_comments=args.post_comments,
            create_epic=args.create_epic,
            output_format=args.output,
            max_parallel=args.max_parallel,
            resume=args.resume,
        )
        return 0
    except KeyboardInterrupt:
        log("INFO", "Interrupted by user")
        return 130
    except Exception as e:
        log("ERROR", f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
