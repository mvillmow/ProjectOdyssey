#!/usr/bin/env python3
"""
Migrate Notes to GitHub Issues

This script migrates documentation from notes/issues/ directories to their
respective GitHub issues as comments. It also creates new issues for
non-numeric directories.

Usage:
    python migrate_notes_to_github.py --dry-run    # Show what would be done
    python migrate_notes_to_github.py --resume     # Resume from saved state
    python migrate_notes_to_github.py              # Actually migrate all

Features:
    - Dry-run mode to preview changes
    - Resume capability for interrupted migrations
    - Rate limiting to respect GitHub API limits
    - Creates new issues for non-numeric directories
    - Detailed logging and progress tracking
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"

    @staticmethod
    def disable():
        """Disable colors for non-terminal output"""
        Colors.HEADER = Colors.OKBLUE = Colors.OKCYAN = ""
        Colors.OKGREEN = Colors.WARNING = Colors.FAIL = ""
        Colors.ENDC = Colors.BOLD = ""


def get_repo_root() -> Path:
    """Get the repository root directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root")


def check_github_rate_limit() -> Tuple[int, float]:
    """Check GitHub API rate limit status."""
    try:
        result = subprocess.run(
            ["gh", "api", "rate_limit"],
            capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        remaining = data["resources"]["core"]["remaining"]
        reset_time = data["resources"]["core"]["reset"]
        return remaining, reset_time
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        logging.warning(f"Failed to check rate limit: {e}")
        return 100, time.time() + 60


def smart_rate_limit_sleep() -> None:
    """Sleep based on GitHub API rate limit status."""
    remaining, reset_time = check_github_rate_limit()

    if remaining <= 10:
        wait_time = max(0, min(reset_time - time.time(), 60))
        if wait_time > 0:
            logging.warning(f"Rate limit critical ({remaining}), waiting {wait_time:.0f}s")
            time.sleep(wait_time)
    elif remaining <= 100:
        backoff = (100 - remaining) / 20
        logging.info(f"Rate limit low ({remaining}), sleeping {backoff:.2f}s")
        time.sleep(backoff)


def check_issue_exists(issue_number: int) -> bool:
    """Check if a GitHub issue exists."""
    try:
        result = subprocess.run(
            ["gh", "issue", "view", str(issue_number), "--json", "number"],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except subprocess.SubprocessError:
        return False


@dataclass
class MigrationState:
    """Tracks migration progress for resume capability."""
    migrated_issues: List[int] = field(default_factory=list)
    created_issues: Dict[str, int] = field(default_factory=dict)  # dir_name -> issue_number
    skipped_issues: List[int] = field(default_factory=list)
    failed_issues: Dict[int, str] = field(default_factory=dict)  # issue_number -> error
    failed_dirs: Dict[str, str] = field(default_factory=dict)  # dir_name -> error
    started_at: str = ""
    last_updated: str = ""

    def save(self, path: Path) -> None:
        """Save state to file."""
        self.last_updated = datetime.now().isoformat()
        with open(path, "w") as f:
            json.dump({
                "migrated_issues": self.migrated_issues,
                "created_issues": self.created_issues,
                "skipped_issues": self.skipped_issues,
                "failed_issues": self.failed_issues,
                "failed_dirs": self.failed_dirs,
                "started_at": self.started_at,
                "last_updated": self.last_updated,
            }, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "MigrationState":
        """Load state from file."""
        if not path.exists():
            return cls(started_at=datetime.now().isoformat())
        with open(path) as f:
            data = json.load(f)
            return cls(**data)


def post_comment_to_issue(issue_number: int, content: str, dry_run: bool = False) -> bool:
    """Post a comment to a GitHub issue."""
    if dry_run:
        logging.info(f"[DRY-RUN] Would post comment to issue #{issue_number}")
        return True

    try:
        smart_rate_limit_sleep()

        # Write content to temp file for --body-file (handles special chars better)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            result = subprocess.run(
                ["gh", "issue", "comment", str(issue_number), "--body-file", temp_path],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                logging.info(f"Posted comment to issue #{issue_number}")
                return True
            else:
                logging.error(f"Failed to post to #{issue_number}: {result.stderr}")
                return False
        finally:
            os.unlink(temp_path)

    except subprocess.SubprocessError as e:
        logging.error(f"Error posting to #{issue_number}: {e}")
        return False


def create_issue_for_dir(dir_name: str, content: str, dry_run: bool = False) -> Optional[int]:
    """Create a new GitHub issue for a non-numeric directory."""
    title = f"[Migrated] {dir_name}"

    if dry_run:
        logging.info(f"[DRY-RUN] Would create issue: {title}")
        return 99999  # Placeholder for dry-run

    try:
        smart_rate_limit_sleep()

        # Create issue body
        body = f"""## Migrated Documentation

This issue was automatically created during the migration of `notes/issues/` to GitHub issues.

**Original directory**: `notes/issues/{dir_name}/`
**Migration date**: {datetime.now().strftime('%Y-%m-%d')}

---

The content from the original README.md will be posted as a comment below.
"""

        # Write body to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(body)
            body_path = f.name

        try:
            result = subprocess.run(
                [
                    "gh", "issue", "create",
                    "--title", title,
                    "--body-file", body_path,
                    "--label", "migrated-notes"
                ],
                capture_output=True, text=True
            )

            if result.returncode == 0:
                # Extract issue number from URL
                url = result.stdout.strip()
                issue_number = int(url.split("/")[-1])
                logging.info(f"Created issue #{issue_number} for {dir_name}")

                # Post content as comment
                post_comment_to_issue(issue_number, content, dry_run=False)
                return issue_number
            else:
                logging.error(f"Failed to create issue for {dir_name}: {result.stderr}")
                return None
        finally:
            os.unlink(body_path)

    except (subprocess.SubprocessError, ValueError) as e:
        logging.error(f"Error creating issue for {dir_name}: {e}")
        return None


def format_migration_comment(content: str, source_path: str) -> str:
    """Format content as a migration comment."""
    date = datetime.now().strftime("%Y-%m-%d")
    return f"""## Implementation Notes (Migrated from Repository)

{content}

---
*Migrated from `{source_path}` on {date}*
"""


def migrate_notes(
    notes_dir: Path,
    state: MigrationState,
    dry_run: bool = False,
    limit: Optional[int] = None
) -> MigrationState:
    """Migrate all notes to GitHub issues."""

    # Get all directories
    all_dirs = sorted([d for d in notes_dir.iterdir() if d.is_dir()])
    numeric_dirs = [d for d in all_dirs if d.name.isdigit()]
    non_numeric_dirs = [d for d in all_dirs if not d.name.isdigit()]

    print(f"\n{Colors.HEADER}Migration Summary:{Colors.ENDC}")
    print(f"  Total directories: {len(all_dirs)}")
    print(f"  Numeric (existing issues): {len(numeric_dirs)}")
    print(f"  Non-numeric (new issues): {len(non_numeric_dirs)}")
    print(f"  Already migrated: {len(state.migrated_issues)}")
    print(f"  Already created: {len(state.created_issues)}")

    if dry_run:
        print(f"\n{Colors.WARNING}DRY-RUN MODE - No changes will be made{Colors.ENDC}\n")

    # Process numeric directories (post to existing issues)
    print(f"\n{Colors.OKBLUE}Processing numeric directories...{Colors.ENDC}")
    numeric_to_process = [
        d for d in numeric_dirs
        if int(d.name) not in state.migrated_issues
        and int(d.name) not in state.skipped_issues
        and int(d.name) not in state.failed_issues
    ]

    if limit:
        numeric_to_process = numeric_to_process[:limit]

    iterator = tqdm(numeric_to_process, desc="Migrating") if HAS_TQDM else numeric_to_process

    for issue_dir in iterator:
        issue_number = int(issue_dir.name)
        readme_path = issue_dir / "README.md"

        if not readme_path.exists():
            logging.warning(f"No README.md in {issue_dir.name}, skipping")
            state.skipped_issues.append(issue_number)
            continue

        # Check if issue exists
        if not dry_run and not check_issue_exists(issue_number):
            logging.warning(f"Issue #{issue_number} doesn't exist on GitHub, skipping")
            state.skipped_issues.append(issue_number)
            continue

        # Read content
        content = readme_path.read_text()
        formatted = format_migration_comment(content, f"notes/issues/{issue_number}/README.md")

        # Post comment
        if post_comment_to_issue(issue_number, formatted, dry_run=dry_run):
            state.migrated_issues.append(issue_number)
        else:
            state.failed_issues[issue_number] = "Failed to post comment"

        # Save state periodically
        if len(state.migrated_issues) % 10 == 0:
            state_path = get_repo_root() / "logs" / ".migration_state.json"
            state.save(state_path)

    # Process non-numeric directories (create new issues)
    print(f"\n{Colors.OKBLUE}Processing non-numeric directories...{Colors.ENDC}")
    non_numeric_to_process = [
        d for d in non_numeric_dirs
        if d.name not in state.created_issues
        and d.name not in state.failed_dirs
    ]

    for issue_dir in non_numeric_to_process:
        readme_path = issue_dir / "README.md"

        if not readme_path.exists():
            logging.warning(f"No README.md in {issue_dir.name}, skipping")
            state.failed_dirs[issue_dir.name] = "No README.md"
            continue

        content = readme_path.read_text()
        formatted = format_migration_comment(content, f"notes/issues/{issue_dir.name}/README.md")

        issue_number = create_issue_for_dir(issue_dir.name, formatted, dry_run=dry_run)
        if issue_number:
            state.created_issues[issue_dir.name] = issue_number
        else:
            state.failed_dirs[issue_dir.name] = "Failed to create issue"

    return state


def ensure_label_exists(label: str = "migrated-notes") -> bool:
    """Ensure the migration label exists."""
    try:
        # Check if label exists
        result = subprocess.run(
            ["gh", "label", "list", "--search", label],
            capture_output=True, text=True
        )

        if label in result.stdout:
            return True

        # Create label
        result = subprocess.run(
            [
                "gh", "label", "create", label,
                "--description", "Documentation migrated from notes/issues/",
                "--color", "5319e7"
            ],
            capture_output=True, text=True
        )
        return result.returncode == 0

    except subprocess.SubprocessError:
        return False


def print_summary(state: MigrationState) -> None:
    """Print migration summary."""
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}Migration Complete{Colors.ENDC}")
    print(f"{'='*60}")
    print(f"\n{Colors.OKGREEN}Successfully migrated:{Colors.ENDC}")
    print(f"  - Issues with comments added: {len(state.migrated_issues)}")
    print(f"  - New issues created: {len(state.created_issues)}")

    if state.skipped_issues:
        print(f"\n{Colors.WARNING}Skipped (issue doesn't exist):{Colors.ENDC}")
        print(f"  - {len(state.skipped_issues)} issues")
        if len(state.skipped_issues) <= 10:
            print(f"    Issues: {state.skipped_issues}")

    if state.failed_issues:
        print(f"\n{Colors.FAIL}Failed:{Colors.ENDC}")
        for issue_num, error in state.failed_issues.items():
            print(f"  - #{issue_num}: {error}")

    if state.failed_dirs:
        print(f"\n{Colors.FAIL}Failed directories:{Colors.ENDC}")
        for dir_name, error in state.failed_dirs.items():
            print(f"  - {dir_name}: {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate notes/issues/ content to GitHub issues"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved state"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of issues to process (for testing)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Disable colors if not a terminal
    if not sys.stdout.isatty():
        Colors.disable()

    repo_root = get_repo_root()
    notes_dir = repo_root / "notes" / "issues"
    logs_dir = repo_root / "logs"
    state_path = logs_dir / ".migration_state.json"

    # Ensure logs directory exists
    logs_dir.mkdir(exist_ok=True)

    # Check notes/issues exists
    if not notes_dir.exists():
        print(f"{Colors.FAIL}Error: {notes_dir} does not exist{Colors.ENDC}")
        sys.exit(1)

    # Load or create state
    if args.resume and state_path.exists():
        state = MigrationState.load(state_path)
        print(f"{Colors.OKCYAN}Resuming from saved state{Colors.ENDC}")
    else:
        state = MigrationState(started_at=datetime.now().isoformat())

    # Ensure label exists (unless dry-run)
    if not args.dry_run:
        if not ensure_label_exists():
            logging.warning("Could not ensure 'migrated-notes' label exists")

    # Run migration
    state = migrate_notes(
        notes_dir,
        state,
        dry_run=args.dry_run,
        limit=args.limit
    )

    # Save final state
    state.save(state_path)

    # Print summary
    print_summary(state)

    if not args.dry_run:
        print(f"\n{Colors.OKCYAN}State saved to: {state_path}{Colors.ENDC}")


if __name__ == "__main__":
    main()
