#!/usr/bin/env python3
"""
Autonomous Mojo repair with Claude Code.

Key properties:
- Parallel agents (one per file)
- Per-file worktrees
- Per-file branch isolation
- Individual PR for each file
- Branches off main
- KISS + DRY enforced

Path Dependencies:
- Requires .claude/ directory in repository root
- Claude runs with --add-dir .claude to access guidelines
- Prompt references .claude/shared/mojo-guidelines.md and mojo-anti-patterns.md
- If .claude/ is missing or --add-dir omitted, Claude cannot access guidelines
"""

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from datetime import datetime, timezone

# Add scripts/utils to path for retry module
sys.path.insert(0, str(Path(__file__).parent))
from utils.retry import retry_with_backoff

# ---------------- Configuration ----------------

MAX_WORKERS_DEFAULT = 6
BUILD_TIMEOUT = 120
# Adaptive timeout: start with 300s for fast failure, increase on retry
CLAUDE_TIMEOUTS = [300, 600, 900]  # Progressive timeouts for retry

BASE_BRANCH = "main"
REMOTE = "origin"

ROOT = Path.cwd()
WORKTREE_BASE = ROOT / "worktrees"
LOG_DIR = ROOT / "build" / "logs"

print_lock = threading.Lock()
stop_processing = threading.Event()
dry_run = False  # Set via --dry-run flag
verbose = False  # Set via --verbose flag

# Metrics tracking
metrics_lock = threading.Lock()
metrics = {
    "start_time": None,
    "end_time": None,
    "files_processed": 0,
    "files_succeeded": 0,
    "files_failed": 0,
    "files_skipped": 0,
    "total_time_seconds": 0.0,
    "retries": {
        "claude_timeouts": 0,
        "git_operations": 0,
    },
    "per_file": [],  # List of {file, success, time_seconds, retries}
}


def ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str):
    with print_lock:
        print(msg, flush=True)


def write_log(path: Path, msg: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts()}] {msg.rstrip()}\n")


def track_file_metrics(file, success, elapsed_time, retry_count=0):
    """Record metrics for a processed file."""
    with metrics_lock:
        metrics["files_processed"] += 1
        if success:
            metrics["files_succeeded"] += 1
        else:
            metrics["files_failed"] += 1

        metrics["per_file"].append(
            {
                "file": file,
                "success": success,
                "time_seconds": round(elapsed_time, 2),
                "retries": retry_count,
            }
        )


def track_retry(retry_type):
    """Track a retry event (claude_timeouts or git_operations)."""
    with metrics_lock:
        if retry_type in metrics["retries"]:
            metrics["retries"][retry_type] += 1


def save_metrics():
    """Save metrics to JSON file."""
    metrics_file = LOG_DIR / "metrics.json"

    with metrics_lock:
        # Calculate derived metrics
        if metrics["files_processed"] > 0:
            metrics["success_rate"] = round(metrics["files_succeeded"] / metrics["files_processed"], 3)
            metrics["average_time_per_file"] = round(
                sum(f["time_seconds"] for f in metrics["per_file"]) / metrics["files_processed"], 2
            )
        else:
            metrics["success_rate"] = 0.0
            metrics["average_time_per_file"] = 0.0

        # Write to file
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

    log(f"📊 Metrics saved to {metrics_file}")


def run(cmd, cwd=None, timeout=None):
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


@retry_with_backoff(max_retries=3, initial_delay=2.0, logger=log)
def run_with_retry(cmd, cwd=None, timeout=None):
    """Run command with automatic retry on network errors.

    Wraps run() with retry logic for transient failures like:
    - Network timeouts
    - Connection errors
    - Rate limiting
    - Temporary Git server issues

    Args:
        cmd: Command to execute
        cwd: Working directory
        timeout: Timeout in seconds

    Returns:
        CompletedProcess result

    Raises:
        Exception after max retries exhausted
    """
    result = run(cmd, cwd=cwd, timeout=timeout)

    # Treat non-zero exit codes as errors for retry logic
    if result.returncode != 0:
        # Check if it's a network error worth retrying
        error_msg = result.stderr.lower() if result.stderr else ""
        if any(
            keyword in error_msg
            for keyword in [
                "connection",
                "network",
                "timeout",
                "temporary failure",
                "could not resolve",
            ]
        ):
            raise ConnectionError(f"Network error: {result.stderr}")

    return result


def check_dependencies():
    """Verify required external dependencies are available.

    Raises:
        RuntimeError: If any required command is missing.
    """
    required = ["mojo", "pixi", "gh", "git", "claude"]
    missing = []

    for cmd in required:
        if not shutil.which(cmd):
            missing.append(cmd)

    if missing:
        raise RuntimeError(
            f"Required command(s) not found: {', '.join(missing)}\n"
            f"Please install missing dependencies before running this script."
        )

    # Verify gh CLI is authenticated
    result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("GitHub CLI (gh) is not authenticated.\nRun 'gh auth login' to authenticate.")


def health_check():
    """Check and display status of all required dependencies.

    Displays version information and availability status for:
    - mojo (Mojo compiler)
    - pixi (Package manager)
    - gh (GitHub CLI)
    - git (Version control)
    - claude (Claude Code CLI)

    Exit codes:
        0: All dependencies available and working
        1: One or more dependencies missing or non-functional
    """
    required = {
        "mojo": ["mojo", "--version"],
        "pixi": ["pixi", "--version"],
        "gh": ["gh", "--version"],
        "git": ["git", "--version"],
        "claude": ["claude", "--version"],
    }

    log("=" * 80)
    log("DEPENDENCY HEALTH CHECK")
    log("=" * 80)
    log("")

    all_ok = True
    for cmd, version_cmd in required.items():
        # Check if command exists
        cmd_path = shutil.which(cmd)

        if not cmd_path:
            log(f"✗ {cmd:12} NOT FOUND")
            all_ok = False
            continue

        # Get version information
        try:
            result = subprocess.run(version_cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Extract first line of version output
                version = result.stdout.strip().split("\n")[0]
                log(f"✓ {cmd:12} {version}")
            else:
                log(f"⚠ {cmd:12} FOUND but version check failed")
                all_ok = False
        except Exception as e:
            log(f"⚠ {cmd:12} FOUND but error getting version: {e}")
            all_ok = False

    log("")
    log("-" * 80)
    log("GitHub CLI Authentication")
    log("-" * 80)

    # Check gh auth status
    try:
        result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Parse auth status output
            auth_lines = result.stderr.strip().split("\n")  # gh outputs to stderr
            log("✓ GitHub CLI authenticated")
            for line in auth_lines[:3]:  # Show first 3 lines
                log(f"  {line}")
        else:
            log("✗ GitHub CLI NOT authenticated")
            log("  Run 'gh auth login' to authenticate")
            all_ok = False
    except Exception as e:
        log(f"⚠ Error checking gh auth: {e}")
        all_ok = False

    log("")
    log("=" * 80)
    if all_ok:
        log("✓ ALL DEPENDENCIES OK")
        log("=" * 80)
        return 0
    else:
        log("✗ SOME DEPENDENCIES MISSING OR NON-FUNCTIONAL")
        log("=" * 80)
        return 1


# ---------------- Build ----------------


def build_cmd(root, file):
    return [
        "pixi",
        "run",
        "mojo",
        "build",
        "-g",
        "--no-optimization",
        "--validate-doc-strings",
        "-I",
        root,
        file,
        "-o",
        f"build/debug/{Path(file).stem}",
    ]


def build_ok(cwd, root, file, log_path) -> bool:
    write_log(log_path, f"Building: {file}")

    if verbose:
        cmd_str = " ".join(str(c) for c in build_cmd(root, file))
        log(f"  Running: {cmd_str}")

    r = run(build_cmd(root, file), cwd=cwd, timeout=BUILD_TIMEOUT)

    write_log(log_path, "Build output:")
    write_log(log_path, r.stderr if r.stderr else "(no output)")
    write_log(log_path, f"Build exit code: {r.returncode}")

    if verbose and r.stderr:
        # Show first 10 lines of build errors/warnings
        error_lines = r.stderr.split("\n")[:10]
        log("  Build errors/warnings (first 10 lines):")
        for line in error_lines:
            log(f"    {line}")

    # Check for warnings in stderr (case-insensitive)
    has_warnings = False
    if r.stderr and "warning:" in r.stderr.lower():
        has_warnings = True
        write_log(log_path, "BUILD HAS WARNINGS")

    if r.returncode == 0 and not has_warnings:
        write_log(log_path, "BUILD OK (no errors, no warnings)")
        return True

    if has_warnings:
        write_log(log_path, "BUILD FAILED (warnings present)")
        if verbose:
            log("  ✗ Build has warnings - must be fixed")
    else:
        write_log(log_path, f"BUILD FAILED (exit={r.returncode})")

    return False


# ---------------- Git helpers ----------------


def sanitize_branch_name(file_path: str) -> str:
    """Convert file path to valid branch name."""
    # Remove extension and leading ./
    name = Path(file_path).with_suffix("").as_posix()
    if name.startswith("./"):
        name = name[2:]
    # Replace slashes and special chars with hyphens
    name = name.replace("/", "-").replace("_", "-")
    # Remove any remaining invalid characters
    name = "".join(c if c.isalnum() or c == "-" else "" for c in name)
    # Ensure it starts with a letter (prepend 'fix-' if needed)
    if not name[0].isalpha():
        name = f"fix-{name}"
    return f"fix-{name}"


def ensure_base_branch():
    run(["git", "fetch", REMOTE])
    r = run(["git", "show-ref", "--verify", f"refs/remotes/{REMOTE}/{BASE_BRANCH}"])
    if r.returncode != 0:
        raise RuntimeError(f"{REMOTE}/{BASE_BRANCH} does not exist")


def create_worktree(branch):
    """Create worktree branching from latest main."""
    path = WORKTREE_BASE / branch
    if path.exists():
        if verbose:
            log(f"  Removing existing worktree directory: {path}")
        shutil.rmtree(path, ignore_errors=True)

    # Fetch latest main before creating worktree
    if verbose:
        log(f"  Fetching latest {REMOTE}/{BASE_BRANCH}")
    r = run(["git", "fetch", REMOTE, BASE_BRANCH])
    if r.returncode != 0:
        raise RuntimeError(f"Failed to fetch {REMOTE}/{BASE_BRANCH}: {r.stderr}")

    # Delete existing branch if it exists (git branch -D is idempotent)
    if verbose:
        log(f"  Deleting branch {branch} if it exists")
    run(["git", "branch", "-D", branch])  # Ignore errors - branch may not exist

    # Create worktree with new branch
    if verbose:
        log(f"  Creating worktree at {path} with branch {branch}")
    r = run(
        [
            "git",
            "worktree",
            "add",
            "--force",
            "-b",
            branch,
            str(path),
            f"{REMOTE}/{BASE_BRANCH}",
        ]
    )
    if r.returncode != 0:
        raise RuntimeError(f"Failed to create worktree: {r.stderr}")

    if verbose:
        log(f"  ✓ Worktree created at {path}")

    return path


def cleanup_worktree(path, branch=None):
    """Remove git worktree and optionally delete remote branch.

    Args:
        path: Path to worktree directory
        branch: Branch name to delete from remote (optional)
    """
    run(["git", "worktree", "remove", "--force", path])
    shutil.rmtree(path, ignore_errors=True)

    # Delete remote branch if provided (prevents orphaned branches)
    if branch:
        run(["git", "push", REMOTE, "--delete", branch], check=False)


def ensure_clean_git(cwd, log_path) -> bool:
    r = run(["git", "status", "--porcelain"], cwd=cwd)
    if r.stdout.strip():
        write_log(log_path, "DIRTY WORKTREE — aborting")
        return False
    return True


def has_commit(cwd) -> bool:
    r = run(["git", "rev-parse", "--verify", "HEAD"], cwd=cwd)
    return r.returncode == 0


def rebase_on_main(cwd, log_path) -> bool:
    """Rebase current branch on latest main."""
    # Fetch latest main with retry (network operation)
    try:
        run_with_retry(["git", "fetch", REMOTE, BASE_BRANCH], cwd=cwd)
    except Exception as e:
        write_log(log_path, f"FETCH FAILED after retries - {e}")
        return False

    # Rebase on main (local operation, no retry needed)
    r = run(["git", "rebase", f"{REMOTE}/{BASE_BRANCH}"], cwd=cwd)
    if r.returncode == 0:
        write_log(log_path, "REBASE OK - rebased on latest main")
        return True

    write_log(log_path, f"REBASE FAILED - {r.stderr}")
    # Abort the rebase
    run(["git", "rebase", "--abort"], cwd=cwd)
    return False


def push_branch(branch, cwd, log_path) -> bool:
    """Push branch to remote with retry on network errors."""
    try:
        r = run_with_retry(["git", "push", "-u", REMOTE, branch], cwd=cwd)
        if r.returncode == 0:
            write_log(log_path, f"PUSH OK - {branch} pushed to {REMOTE}")
            return True
        write_log(log_path, f"PUSH FAILED - {r.stderr}")
        return False
    except Exception as e:
        write_log(log_path, f"PUSH FAILED after retries - {e}")
        return False


def create_pr(branch, file, cwd, log_path) -> bool:
    """Create pull request for the branch and enable auto-merge."""
    if dry_run:
        write_log(log_path, f"DRY RUN - Would create PR for {branch} with auto-merge")
        return True

    # Get commit message to use as PR title
    r = run(["git", "log", "-1", "--format=%s"], cwd=cwd)
    title = r.stdout.strip() if r.returncode == 0 else f"fix: {file}"

    # Get commit body for PR description
    r = run(["git", "log", "-1", "--format=%b"], cwd=cwd)
    body = r.stdout.strip() if r.returncode == 0 else f"Automated fix for {file}"

    # Create PR using gh CLI with retry
    cmd = [
        "gh",
        "pr",
        "create",
        "--title",
        title,
        "--body",
        body,
        "--base",
        BASE_BRANCH,
        "--head",
        branch,
    ]

    try:
        r = run_with_retry(cmd, cwd=cwd)
        if r.returncode != 0:
            write_log(log_path, f"PR CREATION FAILED - {r.stderr}")
            return False
    except Exception as e:
        write_log(log_path, f"PR CREATION FAILED after retries - {e}")
        return False

    pr_url = r.stdout.strip()
    write_log(log_path, f"PR CREATED - {pr_url}")
    if verbose:
        log(f"✓ PR created for {file}: {pr_url}")

    # Enable auto-merge using rebase strategy with retry
    merge_cmd = ["gh", "pr", "merge", pr_url, "--auto", "--rebase"]
    try:
        r = run_with_retry(merge_cmd, cwd=cwd)
        if r.returncode == 0:
            write_log(log_path, "AUTO-MERGE ENABLED")
            if verbose:
                log(f"✓ Auto-merge enabled for {file}")
            return True

        write_log(log_path, f"AUTO-MERGE FAILED - {r.stderr}")
    except Exception as e:
        write_log(log_path, f"AUTO-MERGE FAILED after retries - {e}")
    if verbose:
        log(f"⚠️  Auto-merge failed for {file}: {r.stderr}")
    # Return True anyway since PR was created
    return True


# ---------------- Claude ----------------


def build_prompt(file: str, root: str) -> str:
    """Build comprehensive Claude 4 prompt for Mojo build fixing."""
    return f"""<task_context>
<role>
You are an autonomous Mojo build repair agent working in parallel with other agents.
Each agent creates a separate pull request for review.
</role>

<why_this_matters>
- Failed builds block the entire CI/CD pipeline
- Incorrect fixes introduce bugs that affect downstream code
- Multiple agents work in parallel - each creates a PR for one file
- PRs will be reviewed before merging to main
</why_this_matters>

<file_to_fix>
{file}
</file_to_fix>

<working_directory>
{root}
</working_directory>
</task_context>

<critical_first_step>
BEFORE doing anything else:
1. Run the build command to see if the file already compiles successfully
2. If it compiles (exit code 0, no errors, no warnings), do NOT make any changes - just exit
3. Proceed with fixes if there are compilation errors OR warnings

IMPORTANT: The build must produce ZERO warnings. Warnings are NOT acceptable and must be fixed.

Build command:
pixi run mojo build -g --no-optimization --validate-doc-strings -I {root} {file} -o build/debug/{Path(file).stem}
</critical_first_step>

<mojo_language_rules>
<reference_documentation>
CRITICAL: Read these files BEFORE making any changes:

1. .claude/shared/mojo-guidelines.md
   - Mojo v0.26.1+ syntax and parameter conventions
   - Constructor patterns (out self, mut self)
   - Ownership transfer (^ operator)
   - List initialization syntax
   - Deprecated patterns to avoid

2. .claude/shared/mojo-anti-patterns.md
   - 64+ test failure patterns from real PRs
   - Ownership violations
   - Constructor signature mistakes
   - Uninitialized data issues
   - Type system gotchas

3. .claude/skills/ directory
   - Mojo-specific skills for common operations
   - validate-mojo-patterns, mojo-format, mojo-lint-syntax
   - Use these skills when appropriate

Read ALL relevant documentation before attempting fixes.
</reference_documentation>
</mojo_language_rules>

<fix_principles>
<code_quality>
- KISS (Keep It Simple): Make the smallest possible fix
  Context: Minimal changes are easier to verify and less likely to introduce bugs

- DRY (Don't Repeat Yourself): Never duplicate code
  Context: Duplication creates maintenance burden and drift

- Preserve Intent: Keep existing APIs and behavior unchanged
  Context: API changes break callers throughout the codebase
</code_quality>

<constraints>
- NEVER delete files (they may be referenced elsewhere)
- NEVER refactor or reorganize (outside scope of build fixes)
- NEVER add features or enhancements (only fix what's broken)
- Work only on the assigned file - don't modify other files
</constraints>
</fix_principles>

<workflow>
<phase name="verify">
<step number="1">
Run build command to check current status
</step>
<step number="2">
If compilation succeeds (exit code 0) AND produces zero warnings, STOP - no fixes needed
</step>
<step number="3">
If compilation fails OR produces warnings, capture and analyze the error/warning output
</step>
</phase>

<phase name="analyze">
<extended_thinking>
After reading the error/warning output, use extended thinking to:
- Identify the root cause of the compilation failure or warning
- Determine if this is a known anti-pattern (check list above)
- Plan the minimal fix that resolves the error/warning
- Consider if there are related errors/warnings in nearby code
- Verify your understanding before making changes
</extended_thinking>
</phase>

<phase name="implement">
<step number="1">
Read the file and understand the context around the error
</step>
<step number="2">
Make the smallest change that fixes the root cause
- Follow Mojo v0.26.1+ syntax patterns
- Avoid known anti-patterns
- Use existing patterns from the codebase
</step>
<step number="3">
Build again to verify the fix works
</step>
<step number="4">
If build still fails, iterate:
- Re-analyze the new error
- Refine your fix
- Try again
- Don't give up until the build succeeds
</step>
</phase>

<phase name="finalize">
<step number="1">
Verify the final build succeeds with ZERO warnings
- Exit code must be 0
- Stderr must contain NO warning messages
- If any warnings remain, go back to analyze phase and fix them
</step>
<step number="2">
Create a git commit with descriptive message:
Format:
fix(module): Brief description

- Root cause: [what was broken]
- Solution: [how it was fixed]
- Patterns used: [any Mojo patterns applied]
</step>
</phase>
</workflow>

<tool_usage_optimization>
<parallel_execution>
When you need to read multiple files, run independent commands, or gather information
from multiple sources, execute these operations IN PARALLEL.

Example: If analyzing an error requires reading 3 related files, make 3 Read tool
calls simultaneously rather than sequentially.

This is CRITICAL for performance - the script runs many files in parallel, and
each agent should maximize its own parallelism.
</parallel_execution>

<reflection_after_tool_use>
After receiving results from ANY tool (especially build output or error messages):
1. Take time to THINK about what the results mean
2. Plan your next action based on the new information
3. Don't rush to the next step without understanding
4. Use extended thinking for complex or unclear situations
</reflection_after_tool_use>
</tool_usage_optimization>

<token_budget_policy>
<ignore_context_limits>
Your context window will be automatically managed. DO NOT stop work early due to
token budget concerns. Even if you're approaching your context limit:
- Continue working until the task is COMPLETE
- Save state to files if needed
- Don't artificially stop just because the end is approaching
- Complete the build fix no matter how many iterations it takes
</ignore_context_limits>
</token_budget_policy>

<success_criteria>
The task is ONLY complete when ALL of these are true:
1. ✅ File compiles successfully (exit code 0)
2. ✅ No compilation errors in stderr
3. ✅ ZERO warnings in stderr (ALL warnings must be fixed, not just avoiding new ones)
4. ✅ Changes are minimal and targeted
5. ✅ Existing APIs and behavior preserved
6. ✅ Git commit created with clear message
7. ✅ Commit message explains root cause and solution

CRITICAL: Do NOT stop until all criteria are met, especially the ZERO warnings requirement.
</success_criteria>

<examples>
<example_1>
Error: "cannot transfer ownership of temporary rvalue"

Root cause: List[Int]() is a temporary, can't transfer to var parameter

Fix:
```mojo
# Before (WRONG)
var tensor = ExTensor(List[Int](), DType.int32)

# After (CORRECT)
var shape = List[Int]()
var tensor = ExTensor(shape, DType.int32)
```

Commit message:
fix(core): resolve ownership transfer error in tensor creation

- Root cause: Temporary rvalue List[Int]() cannot transfer ownership
- Solution: Create named variable for shape before passing to constructor
- Pattern: Ownership transfer to var parameters requires named variable
</example_1>

<example_2>
Error: "fn __init__(mut self, ...) should use out self"

Root cause: Constructor uses deprecated mut self convention

Fix:
```mojo
# Before (WRONG)
fn __init__(mut self, value: Int):
    self.value = value

# After (CORRECT)
fn __init__(out self, value: Int):
    self.value = value
```

Commit message:
fix(layers): update constructor to use out self parameter

- Root cause: Constructor used deprecated mut self convention
- Solution: Changed to out self per Mojo v0.25.7+ guidelines
- Pattern: All __init__ methods must use out self
</example_2>
</examples>
</task_context>"""


def run_claude(prompt, cwd, log_path, timeout=None):
    """Run Claude with the given prompt in the specified working directory.

    Args:
        prompt: Prompt text to send to Claude
        cwd: Working directory for Claude execution
        log_path: Path to log file
        timeout: Timeout in seconds (defaults to CLAUDE_TIMEOUTS[0])
    """
    if timeout is None:
        timeout = CLAUDE_TIMEOUTS[0]

    if dry_run:
        write_log(log_path, "DRY RUN - Would run Claude here")
        write_log(log_path, f"Prompt length: {len(prompt)} chars")
        write_log(log_path, f"Timeout: {timeout}s")
        if verbose:
            log(f"DRY RUN - Would process with {len(prompt)} char prompt, timeout={timeout}s")
        return

    allow_tools = [
        "Read",
        "Edit",
        "Glob",
        "Grep",
        "Web",
        "Bash(gh:*)",
        "Bash(pixi:*)",
        "Bash(mojo:*)",
        "Bash(git status)",
        "Bash(git diff)",
        "Bash(git add:*)",
        "Bash(git commit:*)",
    ]

    # Reference the chief-architect file instead of passing content
    architect_file = Path(".claude/agents/chief-architect.md")
    system_prompt_addition = f"See {architect_file} for chief architect guidance."

    # Build command - pass prompt via stdin to avoid ARG_MAX
    cmd = [
        "claude",
        "-p",
        "--permission-mode",
        "dontAsk",
        "--dangerously-skip-permissions",
        "--append-system-prompt",
        system_prompt_addition,
        "--allowedTools",
        ",".join(allow_tools),
        # CRITICAL: --add-dir allows Claude to access .claude/shared/ files
        # The prompt references mojo-guidelines.md and mojo-anti-patterns.md
        # Without this flag, Claude cannot read these files from the worktree
        "--add-dir",
        ".claude",
    ]

    write_log(log_path, "=" * 80)
    write_log(log_path, "CLAUDE PROMPT (INPUT)")
    write_log(log_path, "=" * 80)
    write_log(log_path, prompt)
    write_log(log_path, "=" * 80)
    write_log(log_path, f"RUNNING CLAUDE (timeout={timeout}s)")
    write_log(log_path, "=" * 80)

    if verbose:
        log(f"Running Claude with {len(prompt)} char prompt in {cwd}, timeout={timeout}s")
        log("Prompt sent:")
        # Show first 500 chars of prompt in console
        log(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        log("-" * 80)

    if verbose:
        log(f"⏳ Claude is processing... (timeout: {timeout}s)")

    # Pass prompt via stdin
    r = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        input=prompt,
        capture_output=True,
        timeout=timeout,
    )

    write_log(log_path, "=" * 80)
    write_log(log_path, f"CLAUDE OUTPUT (exit code: {r.returncode})")
    write_log(log_path, "=" * 80)
    write_log(log_path, "STDOUT:")
    write_log(log_path, r.stdout if r.stdout else "(empty)")
    write_log(log_path, "")
    write_log(log_path, "STDERR:")
    write_log(log_path, r.stderr if r.stderr else "(empty)")
    write_log(log_path, "=" * 80)

    if verbose:
        if r.returncode == 0:
            log(f"✓ Claude completed successfully in {cwd}")
        else:
            log(f"✗ Claude failed (exit={r.returncode}) in {cwd}")

        # Show Claude's actual output in console
        if r.stdout:
            log("Claude's response:")
            log("-" * 80)
            log(r.stdout)
            log("-" * 80)

        if r.stderr:
            log("Claude's errors:")
            log("-" * 80)
            log(r.stderr)
            log("-" * 80)


# ---------------- Worker ----------------


def worker_wrapper(file, root):
    """Wrapper that respects semaphore and stop event."""
    if stop_processing.is_set():
        log(f"Skipping {file} - processing stopped")
        return

    # Don't acquire semaphore in wrapper - let ThreadPoolExecutor handle it
    try:
        process_file(file, root)
    except Exception as e:
        log(f"ERROR processing {file}: {e}")
        if "rate limit" in str(e).lower() or "quota" in str(e).lower():
            log("Claude Code limit hit - stopping all processing")
            stop_processing.set()
            raise


def process_file(file, root):
    """Process a single file with isolated worktree and Claude agent."""
    start_time = time.time()
    retry_count = 0
    success = False

    # Create branch name from file path
    branch = sanitize_branch_name(file)
    log_path = LOG_DIR / f"{branch}.log"

    write_log(log_path, "=" * 80)
    write_log(log_path, f"PROCESSING FILE: {file}")
    write_log(log_path, f"BRANCH: {branch}")
    write_log(log_path, f"TIMESTAMP: {ts()}")
    write_log(log_path, "=" * 80)

    if verbose:
        log(f"[{branch}] Processing {file}")

    wt = create_worktree(branch)
    branch_pushed = False  # Track if remote branch exists
    pr_created = False  # Track if PR was successfully created

    try:
        # CRITICAL: Check if file already compiles
        write_log(log_path, "")
        write_log(log_path, "-" * 80)
        write_log(log_path, "INITIAL BUILD CHECK")
        write_log(log_path, "-" * 80)
        write_log(log_path, "Checking if file compiles...")
        if verbose:
            log(f"[{branch}] Checking build status...")

        if build_ok(wt, root, file, log_path):
            write_log(log_path, "File already compiles - skipping fixes")
            if verbose:
                log(f"[{branch}] ✓ {file} already compiles")
            return

        write_log(log_path, "File has compilation errors - running Claude")
        if verbose:
            log(f"[{branch}] ✗ {file} has errors - fixing...")

        # Build comprehensive prompt
        prompt = build_prompt(file, root)

        # Run Claude with adaptive timeout (retry with increasing timeouts on timeout)
        for attempt, timeout in enumerate(CLAUDE_TIMEOUTS, start=1):
            try:
                write_log(log_path, f"Claude attempt {attempt}/{len(CLAUDE_TIMEOUTS)} (timeout={timeout}s)")
                if verbose and attempt > 1:
                    log(
                        f"[{branch}] Retrying Claude with {timeout}s timeout (attempt {attempt}/{len(CLAUDE_TIMEOUTS)})"
                    )

                run_claude(prompt, wt, log_path, timeout=timeout)
                break  # Success - exit retry loop
            except subprocess.TimeoutExpired:
                write_log(log_path, f"Claude timeout after {timeout}s")
                if verbose:
                    log(f"[{branch}] ⚠ Claude timeout after {timeout}s")

                # Track retry
                retry_count += 1
                track_retry("claude_timeouts")

                if attempt == len(CLAUDE_TIMEOUTS):
                    # All retries exhausted
                    write_log(log_path, "All Claude retries exhausted - abandoning")
                    if verbose:
                        log(f"[{branch}] ✗ Claude timeout even after {timeout}s")
                    return
                # Otherwise, continue to next retry with higher timeout

        # Check if Claude made a commit
        if not has_commit(wt):
            write_log(log_path, "NO COMMIT — Claude did not fix the file")
            if verbose:
                log(f"[{branch}] ✗ No commit made")
            return

        # Verify build passes after fix
        write_log(log_path, "")
        write_log(log_path, "-" * 80)
        write_log(log_path, "POST-CLAUDE BUILD VERIFICATION")
        write_log(log_path, "-" * 80)

        if not build_ok(wt, root, file, log_path):
            write_log(log_path, "Build failed after Claude's fix - abandoning")
            if verbose:
                log(f"[{branch}] ✗ Build still fails after fix")
            return

        # Verify git state is clean
        if not ensure_clean_git(wt, log_path):
            write_log(log_path, "Dirty worktree - abandoning")
            return

        # Rebase on latest main before pushing
        if not rebase_on_main(wt, log_path):
            write_log(log_path, "Failed to rebase on main - abandoning")
            if verbose:
                log(f"[{branch}] ✗ Rebase failed")
            return

        # Verify build still passes after rebase
        if not build_ok(wt, root, file, log_path):
            write_log(log_path, "Build failed after rebase - abandoning")
            if verbose:
                log(f"[{branch}] ✗ Build fails after rebase")
            return

        # Push branch to remote
        if not push_branch(branch, wt, log_path):
            write_log(log_path, "Failed to push branch")
            if verbose:
                log(f"[{branch}] ✗ Failed to push")
            return
        branch_pushed = True  # Track that branch exists on remote

        # Create pull request
        if not create_pr(branch, file, wt, log_path):
            write_log(log_path, "Failed to create PR")
            if verbose:
                log(f"[{branch}] ✗ Failed to create PR")
            return

        write_log(log_path, f"SUCCESS - {file} fixed, PR created")
        if verbose:
            log(f"[{branch}] ✓ {file} fixed and PR created")
        pr_created = True  # Mark success
        success = True  # Track for metrics

    except Exception as e:
        write_log(log_path, f"EXCEPTION: {e}")
        if "rate limit" in str(e).lower() or "quota" in str(e).lower():
            log("⚠️  Claude Code limit hit - stopping all processing")
            stop_processing.set()
            raise
    finally:
        # Only delete remote branch if it was pushed but PR wasn't created
        # (prevents orphaned branches from failed PR creation)
        cleanup_branch = branch if (branch_pushed and not pr_created) else None
        cleanup_worktree(wt, cleanup_branch)

        # Track metrics
        elapsed_time = time.time() - start_time
        track_file_metrics(file, success, elapsed_time, retry_count)


# ---------------- Main ----------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="File containing list of files to process")
    parser.add_argument("--root", "-r", help="Root directory for include paths")
    parser.add_argument("--workers", "-w", type=int, default=MAX_WORKERS_DEFAULT, help="Number of parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Test run without actually calling Claude")
    parser.add_argument("--limit", "-n", type=int, help="Only process first N files (for testing)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--health-check", action="store_true", help="Check dependency status and exit")
    args = parser.parse_args()

    # Health check mode - run and exit
    if args.health_check:
        exit_code = health_check()
        sys.exit(exit_code)

    # Validate required arguments for normal operation
    if not args.input or not args.root:
        parser.error("--input and --root are required (unless using --health-check)")

    # Set global flags
    global dry_run, verbose
    dry_run = args.dry_run
    verbose = args.verbose

    if dry_run:
        log("🔍 DRY RUN MODE - Will not execute Claude or make changes")

    if verbose:
        log("📢 VERBOSE MODE - Detailed logging enabled")

    # Verify all required dependencies are available
    check_dependencies()

    ensure_base_branch()

    # Initialize metrics tracking
    metrics["start_time"] = ts()

    files = Path(args.input).read_text().splitlines()
    WORKTREE_BASE.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Filter out empty lines
        files_to_process = [f for f in files if f.strip()]

        # Apply limit if specified
        if args.limit:
            files_to_process = files_to_process[: args.limit]
            log(f"⚠️  LIMIT MODE - Processing only first {args.limit} files")

        log(f"Processing {len(files_to_process)} files with {args.workers} workers")

        # Submit jobs incrementally with stop-processing check
        futures = []
        for file in files_to_process:
            if stop_processing.is_set():
                log("Stopping - Claude Code limit reached")
                break
            future = pool.submit(worker_wrapper, file, args.root)
            futures.append(future)

        # Wait for completion
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log(f"Worker failed: {e}")

        if stop_processing.is_set():
            log("⚠️  Processing stopped due to Claude Code limit")
        else:
            log(f"✓ Completed processing {len(files_to_process)} files")

    # Finalize and save metrics
    metrics["end_time"] = ts()
    save_metrics()


if __name__ == "__main__":
    main()
