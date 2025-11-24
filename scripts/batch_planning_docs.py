#!/usr/bin/env python3
"""
Batch create planning documentation for multiple issues.

Usage:
    python3 scripts/batch_planning_docs.py <issue_numbers>
"""

import sys
import subprocess
import json
from pathlib import Path
from typing import Dict


def get_issue_details(issue_number: int) -> Dict:
    """Get issue title and body using gh CLI."""
    try:
        result = subprocess.run(
            ["gh", "issue", "view", str(issue_number), "--json", "title,body,labels"],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error getting issue {issue_number}: {e}", file=sys.stderr)
        return None


def extract_phase_from_title(title: str) -> str:
    """Extract phase from issue title."""
    if title.startswith("[Plan]"):
        return "Plan"
    elif title.startswith("[Test]"):
        return "Test"
    elif title.startswith("[Impl]"):
        return "Implementation"
    elif title.startswith("[Package]"):
        return "Package"
    elif title.startswith("[Cleanup]"):
        return "Cleanup"
    return "Unknown"


def extract_component_from_title(title: str) -> str:
    """Extract component name from issue title."""
    # Remove phase prefix
    for prefix in ["[Plan]", "[Test]", "[Impl]", "[Package]", "[Cleanup]"]:
        if title.startswith(prefix):
            title = title[len(prefix) :].strip()
            break

    # Remove phase suffix if present
    for suffix in [
        "- Design and Documentation",
        "- Write Tests",
        "- Implementation",
        "- Integration and Packaging",
        "- Refactor and Finalize",
    ]:
        if title.endswith(suffix):
            title = title[: -len(suffix)].strip()
            break

    return title


def create_planning_doc(issue_number: int, issue_data: Dict) -> bool:
    """Create planning documentation for a single issue."""
    title = issue_data["title"]
    body = issue_data.get("body", "")
    labels = [label["name"] for label in issue_data.get("labels", [])]

    phase = extract_phase_from_title(title)
    component = extract_component_from_title(title)

    # Determine primary label
    phase_labels = {
        "Plan": "planning",
        "Test": "testing",
        "Implementation": "implementation",
        "Package": "packaging",
        "Cleanup": "cleanup",
    }
    primary_label = phase_labels.get(phase, "unknown")

    # Create directory (dynamic path resolution)
    import subprocess

    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
    repo_root = Path(result.stdout.strip())
    issue_dir = repo_root / "notes" / "issues" / str(issue_number)
    issue_dir.mkdir(parents=True, exist_ok=True)

    # Create README.md
    readme_path = issue_dir / "README.md"

    content = f"""# Issue #{issue_number}: {title}

## Objective

{phase} phase for {component}.

## Phase

{phase}

## Labels

- `{primary_label}`
{chr(10).join(f"- `{label}`" for label in labels if label != primary_label)}

## Deliverables

As specified in the issue description.

## Success Criteria

- [ ] All deliverables completed as specified
- [ ] Documentation updated
- [ ] Tests passing (if applicable)
- [ ] Code review completed
- [ ] Changes merged to main

## References

- Issue: https://github.com/modularml/mojo/issues/{issue_number}
- Related planning documentation in `/notes/plan/`
- Agent hierarchy: `/agents/hierarchy.md`

## Implementation Notes

{body if body else "(To be filled during implementation)"}

## Status

Created: 2025-11-16
Status: Pending
"""

    readme_path.write_text(content)
    print(f"Created: {readme_path}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/batch_planning_docs.py <issue_numbers...>")
        sys.exit(1)

    issue_numbers = [int(arg) for arg in sys.argv[1:]]

    created = 0
    failed = 0

    for issue_num in issue_numbers:
        print(f"\nProcessing issue #{issue_num}...")
        issue_data = get_issue_details(issue_num)

        if issue_data:
            if create_planning_doc(issue_num, issue_data):
                created += 1
            else:
                failed += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Created: {created}")
    print(f"  Failed: {failed}")
    print(f"  Total: {created + failed}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
