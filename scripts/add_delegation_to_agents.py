#!/usr/bin/env python3
"""
Add Delegation sections to agent files that should have them.

Adds Delegation sections to:
- Level 1 orchestrators (code-review-orchestrator)
- Level 3 specialists (review specialists)
"""

import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import get_agents_dir

# Standard delegation section for review specialists
REVIEW_SPECIALIST_DELEGATION = """## Delegation

For standard delegation patterns, escalation rules, and skip-level guidelines, see
[delegation-rules.md](../../agents/delegation-rules.md).

### Coordinates With

- [Code Review Orchestrator](./code-review-orchestrator.md) - Receives review assignments, coordinates with other specialists

### Escalates To

- [Code Review Orchestrator](./code-review-orchestrator.md) - When issues fall outside this specialist's scope
"""

# Delegation for code-review-orchestrator
CODE_REVIEW_ORCHESTRATOR_DELEGATION = """## Delegation

For standard delegation patterns, escalation rules, and skip-level guidelines, see
[delegation-rules.md](../../agents/delegation-rules.md).

### Delegates To

- [Algorithm Review Specialist](./algorithm-review-specialist.md) - Mathematical correctness, gradients, numerical stability
- [Architecture Review Specialist](./architecture-review-specialist.md) - System design, modularity, patterns
- [Data Engineering Review Specialist](./data-engineering-review-specialist.md) - Data pipelines, preprocessing, splits
- [Dependency Review Specialist](./dependency-review-specialist.md) - Dependencies, versions, compatibility
- [Documentation Review Specialist](./documentation-review-specialist.md) - Documentation quality and completeness
- [Implementation Review Specialist](./implementation-review-specialist.md) - Code quality, maintainability, patterns
- [Mojo Language Review Specialist](./mojo-language-review-specialist.md) - Mojo-specific features and idioms
- [Paper Review Specialist](./paper-review-specialist.md) - Academic paper quality and standards
- [Performance Review Specialist](./performance-review-specialist.md) - Performance and optimization
- [Research Review Specialist](./research-review-specialist.md) - Research methodology and rigor
- [Safety Review Specialist](./safety-review-specialist.md) - Memory safety and type safety
- [Security Review Specialist](./security-review-specialist.md) - Security vulnerabilities and threats
- [Test Review Specialist](./test-review-specialist.md) - Test quality and coverage

### Coordinates With

- [CI/CD Orchestrator](./cicd-orchestrator.md) - Integration with automated reviews
"""

def add_delegation_section(file_path: Path, delegation_content: str, dry_run: bool = False) -> bool:
    """Add Delegation section to agent file if missing."""
    content = file_path.read_text()

    # Check if Delegation section already exists
    if re.search(r'^## Delegation', content, re.MULTILINE):
        print(f"  ✓ {file_path.name} already has Delegation section")
        return False

    # Find insertion point (before Examples section if it exists, otherwise before final separator)
    examples_match = re.search(r'\n## Examples', content)
    if examples_match:
        insertion_point = examples_match.start()
        new_content = content[:insertion_point] + '\n' + delegation_content + content[insertion_point:]
    else:
        # Look for final separator
        final_separator_match = re.search(r'\n---\n\*[^\*]+\*\s*$', content)
        if final_separator_match:
            insertion_point = final_separator_match.start()
            new_content = content[:insertion_point] + '\n' + delegation_content + content[insertion_point:]
        else:
            # Look for **Configuration File**: pattern
            config_match = re.search(r'\n---\n\n\*\*Configuration File\*\*:', content)
            if config_match:
                insertion_point = config_match.start()
                new_content = content[:insertion_point] + '\n' + delegation_content + content[insertion_point:]
            else:
                # Insert before end
                new_content = content.rstrip() + '\n\n' + delegation_content + '\n'

    if dry_run:
        print(f"  Would add Delegation to {file_path.name}")
        return True
    else:
        file_path.write_text(new_content)
        print(f"  ✅ Added Delegation to {file_path.name}")
        return True

def main():
    import sys

    agents_dir = get_agents_dir()
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("DRY RUN MODE - No files will be modified\n")

    # Review specialists that need Delegation
    review_specialists = [
        "algorithm-review-specialist.md",
        "architecture-review-specialist.md",
        "blog-writer-specialist.md",
        "data-engineering-review-specialist.md",
        "dependency-review-specialist.md",
        "documentation-review-specialist.md",
        "implementation-review-specialist.md",
        "mojo-language-review-specialist.md",
        "paper-review-specialist.md",
        "performance-review-specialist.md",
        "research-review-specialist.md",
        "safety-review-specialist.md",
        "security-review-specialist.md",
        "test-review-specialist.md",
    ]

    modified_count = 0

    # Add to review specialists
    for specialist in review_specialists:
        file_path = agents_dir / specialist
        if file_path.exists():
            if add_delegation_section(file_path, REVIEW_SPECIALIST_DELEGATION, dry_run):
                modified_count += 1

    # Add to code-review-orchestrator
    code_review_path = agents_dir / "code-review-orchestrator.md"
    if code_review_path.exists():
        if add_delegation_section(code_review_path, CODE_REVIEW_ORCHESTRATOR_DELEGATION, dry_run):
            modified_count += 1

    print(f"\n{'Would modify' if dry_run else 'Modified'} {modified_count} files")

if __name__ == "__main__":
    main()
