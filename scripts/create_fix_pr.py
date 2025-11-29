#!/usr/bin/env python3

"""
Script to create and push a fix for the floor_divide edge case test failure.

This script:
1. Creates a feature branch named 'fix-floor-divide-edge'
2. Commits the changes to shared/core/arithmetic.mojo
3. Pushes the branch to origin
4. Creates a pull request linked to issue #2057
"""

import subprocess
import sys


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"[*] {description}")
    print(f"    Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd="/home/mvillmow/ml-odyssey", capture_output=True, text=True, check=True)
        if result.stdout:
            print(f"    Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"    ERROR: Command failed with exit code {e.returncode}")
        if e.stderr:
            print(f"    Stderr: {e.stderr}")
        raise


def main():
    """Main script execution."""
    print("=" * 70)
    print("Floor Division Edge Case Fix - Git Workflow")
    print("=" * 70)
    print()

    # Step 1: Create feature branch
    try:
        run_command(
            ["git", "checkout", "-b", "fix-floor-divide-edge"], "Creating feature branch 'fix-floor-divide-edge'..."
        )
    except subprocess.CalledProcessError:
        # Branch might already exist, try to switch to it
        print("    Branch may already exist, attempting to switch...")
        run_command(["git", "checkout", "fix-floor-divide-edge"], "Switching to existing branch...")

    # Step 2: Stage changes
    run_command(["git", "add", "shared/core/arithmetic.mojo"], "Staging arithmetic.mojo changes...")

    # Step 3: Commit
    commit_message = """fix(arithmetic): Handle division by zero in floor_divide operation

Floor division now correctly returns infinity for floating-point division by zero,
following IEEE 754 semantics. This fix prevents undefined behavior from attempting
to convert infinity to an integer.

Changes:
- Added @parameter if T.is_floating_point() check in _floor_div_op
- Returns x / y directly when y == 0 to let hardware handle inf/nan
- Updated docstring with IEEE 754 division by zero behavior

Fixes test_floor_divide_edge_cases assertion: 'x // 0 should be inf'

Generated with Claude Code (Senior Implementation Engineer)

Co-Authored-By: Claude <noreply@anthropic.com>"""

    run_command(["git", "commit", "-m", commit_message], "Committing fix...")

    # Step 4: Push branch
    run_command(["git", "push", "-u", "origin", "fix-floor-divide-edge"], "Pushing branch to origin...")

    # Step 5: Create PR
    pr_title = "fix(arithmetic): Handle division by zero in floor_divide"
    pr_body = "Closes #2057"

    try:
        result = run_command(["gh", "pr", "create", "--title", pr_title, "--body", pr_body], "Creating pull request...")
        print()
        print("=" * 70)
        print("SUCCESS: Pull request created!")
        print("=" * 70)
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 70)
        print("WARNING: PR creation may have failed")
        print("=" * 70)
        print(f"Error: {e.stderr}")
        print()
        print("You can manually create the PR with:")
        print(f"  gh pr create --title '{pr_title}' --body '{pr_body}'")
        sys.exit(1)

    print()
    print("=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Monitor CI status: gh pr checks (after ~30 seconds)")
    print("2. Review PR at: https://github.com/mvillmow/ml-odyssey/pulls")
    print("3. Address any review comments")
    print("=" * 70)


if __name__ == "__main__":
    main()
