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
"""

import argparse
import concurrent.futures
import shutil
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timezone

# ---------------- Configuration ----------------

MAX_WORKERS_DEFAULT = 6
BUILD_TIMEOUT = 120
CLAUDE_TIMEOUT = 900

BASE_BRANCH = "main"
REMOTE = "origin"

ROOT = Path.cwd()
WORKTREE_BASE = ROOT / "worktrees"
LOG_DIR = ROOT / "build" / "logs"

print_lock = threading.Lock()
stop_processing = threading.Event()
dry_run = False  # Set via --dry-run flag
verbose = False  # Set via --verbose flag


def ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str):
    with print_lock:
        print(msg, flush=True)


def write_log(path: Path, msg: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts()}] {msg.rstrip()}\n")


def run(cmd, cwd=None, timeout=None):
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


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


def cleanup_worktree(path):
    run(["git", "worktree", "remove", "--force", path])
    shutil.rmtree(path, ignore_errors=True)


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
    # Fetch latest main
    run(["git", "fetch", REMOTE, BASE_BRANCH], cwd=cwd)

    # Rebase on main
    r = run(["git", "rebase", f"{REMOTE}/{BASE_BRANCH}"], cwd=cwd)
    if r.returncode == 0:
        write_log(log_path, "REBASE OK - rebased on latest main")
        return True

    write_log(log_path, f"REBASE FAILED - {r.stderr}")
    # Abort the rebase
    run(["git", "rebase", "--abort"], cwd=cwd)
    return False


def push_branch(branch, cwd, log_path) -> bool:
    """Push branch to remote."""
    r = run(["git", "push", "-u", REMOTE, branch], cwd=cwd)
    if r.returncode == 0:
        write_log(log_path, f"PUSH OK - {branch} pushed to {REMOTE}")
        return True
    write_log(log_path, f"PUSH FAILED - {r.stderr}")
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

    # Create PR using gh CLI
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

    r = run(cmd, cwd=cwd)
    if r.returncode != 0:
        write_log(log_path, f"PR CREATION FAILED - {r.stderr}")
        return False

    pr_url = r.stdout.strip()
    write_log(log_path, f"PR CREATED - {pr_url}")
    if verbose:
        log(f"✓ PR created for {file}: {pr_url}")

    # Enable auto-merge using rebase strategy
    merge_cmd = ["gh", "pr", "merge", pr_url, "--auto", "--rebase"]
    r = run(merge_cmd, cwd=cwd)
    if r.returncode == 0:
        write_log(log_path, "AUTO-MERGE ENABLED")
        if verbose:
            log(f"✓ Auto-merge enabled for {file}")
        return True

    write_log(log_path, f"AUTO-MERGE FAILED - {r.stderr}")
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


def run_claude(prompt, cwd, log_path):
    """Run Claude with the given prompt in the specified working directory."""
    if dry_run:
        write_log(log_path, "DRY RUN - Would run Claude here")
        write_log(log_path, f"Prompt length: {len(prompt)} chars")
        if verbose:
            log(f"DRY RUN - Would process with {len(prompt)} char prompt")
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
        "--add-dir",
        ".claude",
    ]

    write_log(log_path, "=" * 80)
    write_log(log_path, "CLAUDE PROMPT (INPUT)")
    write_log(log_path, "=" * 80)
    write_log(log_path, prompt)
    write_log(log_path, "=" * 80)
    write_log(log_path, "RUNNING CLAUDE")
    write_log(log_path, "=" * 80)

    if verbose:
        log(f"Running Claude with {len(prompt)} char prompt in {cwd}")
        log("Prompt sent:")
        # Show first 500 chars of prompt in console
        log(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        log("-" * 80)

    if verbose:
        log("⏳ Claude is processing... (this may take up to 15 minutes)")

    # Pass prompt via stdin
    r = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        input=prompt,
        capture_output=True,
        timeout=CLAUDE_TIMEOUT,
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

        # Run Claude with enhanced prompt
        run_claude(prompt, wt, log_path)

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

        # Create pull request
        if not create_pr(branch, file, wt, log_path):
            write_log(log_path, "Failed to create PR")
            if verbose:
                log(f"[{branch}] ✗ Failed to create PR")
            return

        write_log(log_path, f"SUCCESS - {file} fixed, PR created")
        if verbose:
            log(f"[{branch}] ✓ {file} fixed and PR created")

    except Exception as e:
        write_log(log_path, f"EXCEPTION: {e}")
        if "rate limit" in str(e).lower() or "quota" in str(e).lower():
            log("⚠️  Claude Code limit hit - stopping all processing")
            stop_processing.set()
            raise
    finally:
        cleanup_worktree(wt)


# ---------------- Main ----------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="File containing list of files to process")
    parser.add_argument("--root", "-r", required=True, help="Root directory for include paths")
    parser.add_argument("--workers", "-w", type=int, default=MAX_WORKERS_DEFAULT, help="Number of parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Test run without actually calling Claude")
    parser.add_argument("--limit", "-n", type=int, help="Only process first N files (for testing)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
