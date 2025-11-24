#!/usr/bin/env python3
"""
Create GitHub issues for a single component for testing/verification.

Usage:
    python scripts/create_single_component_issues.py <path-to-github_issue.md>

Example:
    python scripts/create_single_component_issues.py notes/plan/01-foundation/01-directory-structure/01-create-papers-dir/01-create-base-dir/github_issue.md
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common import LABEL_COLORS


def parse_github_issue_file(file_path):
    """Parse a github_issue.md file and extract all 5 issues."""
    with open(file_path, 'r') as f:
        content = f.read()

    issues = []

    # Split into sections by "## <Type> Issue"
    issue_sections = re.split(r'\n---\n', content)

    issue_types = ['Plan', 'Test', 'Implementation', 'Packaging', 'Cleanup']

    for section in issue_sections:
        # Check which type this is
        for issue_type in issue_types:
            if f'## {issue_type} Issue' in section:
                # Extract title
                title_match = re.search(r'\*\*Title\*\*:\s*`([^`]+)`', section)
                # Extract labels
                labels_match = re.search(r'\*\*Labels\*\*:\s*`([^`]+)`', section)
                # Extract body (everything after "**Body**:" until "**GitHub Issue URL**:")
                body_match = re.search(r'\*\*Body\*\*:\s*\n\n(.*?)\n\n\*\*GitHub Issue URL\*\*:', section, re.DOTALL)

                if title_match and labels_match and body_match:
                    issues.append({
                        'type': issue_type,
                        'title': title_match.group(1).strip(),
                        'labels': labels_match.group(1).strip(),
                        'body': body_match.group(1).strip()
                    })
                break

    return issues


def get_repo_name():
    """Auto-detect repository name from git."""
    try:
        result = subprocess.run(
            ['gh', 'repo', 'view', '--json', 'nameWithOwner', '-q', '.nameWithOwner'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'mvillmow/ml-odyssey'  # Fallback


def create_label_if_needed(label, repo=None):
    """Create a label if it doesn't exist."""
    if repo is None:
        repo = get_repo_name()
    color = LABEL_COLORS.get(label, '000000')

    # Try to create label (will fail silently if it exists)
    cmd = ['gh', 'label', 'create', label, '--color', color, '--repo', repo]
    subprocess.run(cmd, capture_output=True, text=True)


def create_github_issue(title, labels, body, repo=None):
    """Create a single GitHub issue using gh CLI."""
    if repo is None:
        repo = get_repo_name()
    # Ensure labels exist
    for label in labels.split(','):
        create_label_if_needed(label.strip(), repo)

    # Write body to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(body)
        temp_file = f.name

    try:
        # Create issue
        cmd = [
            'gh', 'issue', 'create',
            '--title', title,
            '--body-file', temp_file,
            '--label', labels,
            '--repo', repo
        ]

        print(f"\nCreating issue: {title}")
        print(f"Labels: {labels}")
        print(f"Body length: {len(body)} characters")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Extract issue URL from output
            issue_url = result.stdout.strip()
            print(f"✅ Created: {issue_url}")
            return issue_url
        else:
            print("❌ Failed to create issue")
            print(f"Error: {result.stderr}")
            return None

    finally:
        # Clean up temp file
        Path(temp_file).unlink()


def update_github_issue_file(file_path, issue_urls):
    """Update the github_issue.md file with created issue URLs."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Map issue types to their URLs
    issue_map = {
        'Plan': issue_urls[0] if len(issue_urls) > 0 else None,
        'Test': issue_urls[1] if len(issue_urls) > 1 else None,
        'Implementation': issue_urls[2] if len(issue_urls) > 2 else None,
        'Packaging': issue_urls[3] if len(issue_urls) > 3 else None,
        'Cleanup': issue_urls[4] if len(issue_urls) > 4 else None,
    }

    # Update each issue URL
    for issue_type, url in issue_map.items():
        if url:
            # Pattern to find and replace the URL line for this issue type
            pattern = rf'(## {issue_type} Issue.*?\*\*GitHub Issue URL\*\*:)\s*\[To be created\]'
            replacement = rf'\1 {url}'
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Write back
    with open(file_path, 'w') as f:
        f.write(content)

    print(f"\n✅ Updated {file_path} with issue URLs")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/create_single_component_issues.py <path-to-github_issue.md>")
        print("\nExample:")
        print("  python scripts/create_single_component_issues.py notes/plan/01-foundation/01-directory-structure/01-create-papers-dir/01-create-base-dir/github_issue.md")
        sys.exit(1)

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Processing: {file_path}")
    print("=" * 60)

    # Parse issues
    issues = parse_github_issue_file(file_path)

    if not issues:
        print("Error: No issues found in file")
        sys.exit(1)

    print(f"Found {len(issues)} issues to create")

    # Create each issue
    issue_urls = []
    for i, issue in enumerate(issues, 1):
        print(f"\n[{i}/{len(issues)}] {issue['type']} Issue")
        url = create_github_issue(
            title=issue['title'],
            labels=issue['labels'],
            body=issue['body']
        )
        if url:
            issue_urls.append(url)
        else:
            print(f"⚠️  Failed to create {issue['type']} issue")

    # Update file with URLs
    if issue_urls:
        update_github_issue_file(file_path, issue_urls)

    print("\n" + "=" * 60)
    print(f"Summary: Created {len(issue_urls)}/{len(issues)} issues")
    print("\nIssue URLs:")
    for i, url in enumerate(issue_urls, 1):
        print(f"  {i}. {url}")


if __name__ == '__main__':
    main()
