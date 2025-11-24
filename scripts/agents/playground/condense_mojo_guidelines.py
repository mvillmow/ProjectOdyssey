#!/usr/bin/env python3
"""
Condense Mojo-Specific Guidelines sections in agent files.

Keep full sections only in:
- mojo-language-review-specialist.md (Mojo-focused role)
- chief-architect.md (needs language selection guidance)

For implementation-focused roles, keep brief guidelines.
For other roles, replace with short reference.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common import get_agents_dir

# Files that should keep full Mojo sections
KEEP_FULL = {
    "mojo-language-review-specialist.md",
    "chief-architect.md",
}

# Files that should keep brief guidelines (implementation-focused)
KEEP_BRIEF = {
    "implementation-engineer.md",
    "senior-implementation-engineer.md",
    "junior-implementation-engineer.md",
    "implementation-specialist.md",
    "test-engineer.md",
    "senior-test-engineer.md",
    "junior-test-engineer.md",
    "test-specialist.md",
}

# Short reference for non-implementation roles
SHORT_REFERENCE = """## Language Guidelines

When working with Mojo code, follow patterns in [mojo-language-review-specialist.md](./mojo-language-review-specialist.md). Key principles: prefer `fn` over `def`, use `owned`/`borrowed` for memory safety, leverage SIMD for performance-critical code."""

# Brief guidelines for implementation roles (keep it concise)
BRIEF_GUIDELINES = """## Mojo-Specific Guidelines

### Function Definitions
- Use `fn` for performance-critical code (compile-time checks, optimization)
- Use `def` for prototyping or Python interop
- Default to `fn` unless flexibility is needed

### Memory Management
- Use `owned` for ownership transfer
- Use `borrowed` for read-only access
- Use `inout` for mutable references
- Prefer value semantics (struct) over reference semantics (class)

### Performance
- Leverage SIMD for vectorizable operations
- Use `@parameter` for compile-time constants
- Avoid unnecessary copies with move semantics (`^`)

See [mojo-language-review-specialist.md](./mojo-language-review-specialist.md) for comprehensive guidelines."""


def extract_mojo_section(content: str) -> Tuple[str, str, str]:
    """
    Extract the Mojo-Specific Guidelines section.
    Returns: (before_section, section_content, after_section)
    """
    # Pattern to match from "## Mojo-Specific Guidelines" to the next "## "
    pattern = r"(.*?)(^## Mojo-Specific Guidelines\n\n.*?)(^## )"

    match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1), match.group(2), match.group(3) + content[match.end(3) :]
    return content, "", ""


def condense_mojo_section(filepath: Path) -> Dict:
    """Condense Mojo-Specific Guidelines section based on agent role."""
    result = {"file": filepath.name, "action": "skipped", "lines_removed": 0}

    # Skip files that should keep full sections
    if filepath.name in KEEP_FULL:
        result["action"] = "kept_full"
        return result

    content = filepath.read_text()
    original_content = content

    before, section, after = extract_mojo_section(content)

    if not section:
        result["action"] = "no_section"
        return result

    # Determine replacement based on role
    if filepath.name in KEEP_BRIEF:
        replacement = BRIEF_GUIDELINES + "\n\n"
        result["action"] = "condensed_to_brief"
    else:
        replacement = SHORT_REFERENCE + "\n\n"
        result["action"] = "condensed_to_reference"

    # Reconstruct content
    new_content = before + replacement + after

    if new_content != original_content:
        result["lines_removed"] = len(original_content.splitlines()) - len(new_content.splitlines())
        filepath.write_text(new_content)

    return result


def main() -> None:
    agents_dir = get_agents_dir()

    # Get all agent files with Mojo sections
    agent_files = [
        "agentic-workflows-orchestrator.md",
        "architecture-design.md",
        "chief-architect.md",
        "cicd-orchestrator.md",
        "documentation-engineer.md",
        "documentation-specialist.md",
        "foundation-orchestrator.md",
        "implementation-engineer.md",
        "implementation-specialist.md",
        "integration-design.md",
        "junior-documentation-engineer.md",
        "junior-implementation-engineer.md",
        "junior-test-engineer.md",
        "mojo-language-review-specialist.md",
        "papers-orchestrator.md",
        "performance-engineer.md",
        "performance-specialist.md",
        "security-design.md",
        "security-specialist.md",
        "senior-implementation-engineer.md",
        "shared-library-orchestrator.md",
        "test-engineer.md",
        "test-specialist.md",
        "tooling-orchestrator.md",
    ]

    results = []

    print("Condensing Mojo-Specific Guidelines sections...")
    print("=" * 90)

    for filename in agent_files:
        filepath = agents_dir / filename
        if filepath.exists():
            result = condense_mojo_section(filepath)
            results.append(result)

            if result["action"] == "kept_full":
                print(f"○ {filename:50} | Kept full section")
            elif result["action"] == "no_section":
                print(f"○ {filename:50} | No Mojo section found")
            elif result["action"] == "condensed_to_brief":
                print(f"✓ {filename:50} | Brief guidelines | Lines: {result['lines_removed']:3}")
            elif result["action"] == "condensed_to_reference":
                print(f"✓ {filename:50} | Short reference | Lines: {result['lines_removed']:3}")

    print("=" * 90)
    print("\nSummary:")
    print(f"  Files processed: {len(results)}")
    print(f"  Kept full: {sum(1 for r in results if r['action'] == 'kept_full')}")
    print(f"  Condensed to brief: {sum(1 for r in results if r['action'] == 'condensed_to_brief')}")
    print(f"  Condensed to reference: {sum(1 for r in results if r['action'] == 'condensed_to_reference')}")
    print(f"  No section: {sum(1 for r in results if r['action'] == 'no_section')}")
    print(f"  Total lines removed: {sum(r['lines_removed'] for r in results)}")


if __name__ == "__main__":
    main()
