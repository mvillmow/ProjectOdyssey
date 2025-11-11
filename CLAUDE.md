# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML Odyssey is a Mojo-based AI research platform for reproducing classic research papers. The project uses a
comprehensive 4-level hierarchical planning structure with automated GitHub issue creation.

**Current Status**: Planning phase - repository structure and GitHub issues are being established before implementation begins.

## Working with Agents

This project uses a hierarchical agent system for all development work. **Always use agents** as the primary
method for completing tasks.

### Agent Hierarchy Quick Reference

- **Level 0**: Chief Architect - Strategic decisions and architecture
- **Level 1**: Orchestrators (6) - Section coordination (foundation, shared-library, tooling, first-paper, ci-cd, agentic-workflows)
- **Level 2**: Design/Orchestrators - Module design and code review coordination
- **Level 3**: Specialists (12 review, 3 implementation) - Component-level work
- **Level 4**: Engineers (3) - Implementation tasks
- **Level 5**: Junior Engineers (3) - Simple, well-defined tasks

### Key Agent Principles

1. **Always start with orchestrators** for new section work
2. **All outputs** must go to `/notes/issues/<issue-number>/README.md`
3. **Link all PRs** to issues using `gh pr create --issue <number>` or "Closes #123" in description
4. **Minimal changes only** - smallest change that solves the problem
5. **No scope creep** - focus only on issue requirements
6. **Reply to each review comment** with `✅ Fixed - [brief description]`

### Documentation Rules

- **Issue-specific outputs**: `/notes/issues/<issue-number>/README.md`
- **Comprehensive specs**: `/notes/review/` (architectural decisions, design docs)
- **Team guides**: `/agents/` (quick start, hierarchy, templates)
- **Never duplicate** documentation across locations - link instead

### Language Preference

#### Mojo First - With Pragmatic Exceptions

**Default to Mojo** for ALL ML/AI implementations:

- ✅ Neural network implementations (forward/backward passes, layers)
- ✅ Training loops and optimization algorithms
- ✅ Tensor operations and SIMD kernels
- ✅ Performance-critical data processing
- ✅ Type-safe model components
- ✅ Gradient computation and backpropagation
- ✅ Model inference engines

**Use Python for Automation** when technical limitations require it:

- ✅ Subprocess output capture (Mojo v0.25.7 limitation - cannot capture stdout/stderr)
- ✅ Regex-heavy text processing (no Mojo regex support in stdlib)
- ✅ GitHub API interaction via Python libraries (`gh` CLI, REST API)
- ⚠️ **MUST document justification** (see ADR-001 for header template)

**Rule of Thumb** (Decision Tree):

1. **ML/AI implementation?** → Mojo (required)
2. **Automation needing subprocess output?** → Python (allowed, document why)
3. **Automation needing regex?** → Python (allowed, document why)
4. **Interface with Python-only libraries?** → Python (allowed, document why)
5. **Everything else?** → Mojo (default)

**Why Mojo for ML/AI**:

- Performance: 10-100x faster for ML workloads
- Type safety: Catch errors at compile time
- Memory safety: Built-in ownership and borrow checking
- SIMD optimization: Parallel tensor operations
- Future-proof: Designed for AI/ML from the ground up

**Why Python for Automation**:

- Mojo's subprocess API lacks exit code access (causes silent failures)
- Regex support not production-ready (mojo-regex is alpha stage)
- Python is the right tool for automation - not a temporary workaround

**See**: [ADR-001](notes/review/adr/ADR-001-language-selection-tooling.md) for complete language selection strategy,
technical evidence (test results), and justification requirements

See `/agents/README.md` for complete agent documentation and `/agents/hierarchy.md` for visual hierarchy.

## Environment Setup

This project uses Pixi for environment management:

```bash

# Pixi is already configured - dependencies are in pixi.toml
# Mojo is the primary language target for future implementations

```

## Common Commands

### GitHub Issue Management

```bash

# Test issue creation for a single component
# Note: Plan files are in notes/plan/

python3 scripts/create_single_component_issues.py notes/plan/01-foundation/github_issue.md

# Create issues for one section (recommended approach)

python3 scripts/create_issues.py --section 01-foundation

# Preview what issues would be created without creating them

python3 scripts/create_issues.py --dry-run

# Resume interrupted issue creation

python3 scripts/create_issues.py --resume

# Create all issues (use with caution)

python3 scripts/create_issues.py
```

### Plan File Management

**Note**: Plan files in `notes/plan/` are task-relative and kept locally. They are NOT tracked in version control.
Only tracked documentation (notes/issues/, notes/review/, agents/) should be referenced in commits.

```bash

# Regenerate github_issue.md files from plan.md sources (local files)

python3 scripts/regenerate_github_issues.py

# Preview changes without writing files

python3 scripts/regenerate_github_issues.py --dry-run

# Regenerate one section only

python3 scripts/regenerate_github_issues.py --section 01-foundation
```

### GitHub CLI

```bash

# Check authentication status

gh auth status

# List issues

gh issue list

# View issue details

gh issue view <number>

# Reply to PR review comments (addressing feedback)

gh pr comment <pr-number> --body "Short, concise explanation of what was done"
```

### Handling PR Review Comments

When addressing review comments on a pull request:

1. **Make the requested changes** in your code
2. **Reply to EACH review comment** with a short, concise explanation of what was done
3. **Use the format**: `gh pr comment <pr-number> --body "✅ Fixed - [brief description]"`

**Example responses**:

- `✅ Fixed - Removed unnecessary __init__.py file`
- `✅ Fixed - Deleted script as it's no longer needed`
- `✅ Fixed - Added README.md to papers/ directory`
- `✅ Fixed - Refactored to use list comprehension`

**Important**:

- Keep responses SHORT and CONCISE (1 line preferred)
- Start with ✅ to indicate the issue is resolved
- Explain WHAT was done, not why (unless asked)
- Reply to ALL open review comments

### Agent Testing

Agent configurations are automatically validated in CI on all PRs. Run tests locally before committing:

```bash
# Validate agent YAML frontmatter and configuration
python3 tests/agents/validate_configs.py .claude/agents/

# Test agent discovery and loading
python3 tests/agents/test_loading.py .claude/agents/

# Test delegation patterns
python3 tests/agents/test_delegation.py .claude/agents/

# Test workflow integration
python3 tests/agents/test_integration.py .claude/agents/

# Test Mojo-specific patterns
python3 tests/agents/test_mojo_patterns.py .claude/agents/

# Run all tests
for script in tests/agents/test_*.py tests/agents/validate_*.py; do
    python3 "$script" .claude/agents/
done
```

**Test Coverage**:

- Configuration validation (YAML frontmatter, required fields, tool specifications)
- Agent discovery and loading (hierarchy coverage, activation patterns)
- Delegation patterns (chain validation, escalation paths)
- Workflow integration (5-phase coverage, parallel execution)
- Mojo patterns (fn vs def, struct vs class, SIMD, memory management)

**CI Integration**: The `.github/workflows/test-agents.yml` workflow runs these tests automatically on all PRs
affecting agent configurations.

### Pre-commit Hooks

Pre-commit hooks automatically check code quality before commits. The hooks include `mojo format` for Mojo code and
markdown linting for documentation.

```bash

# Install pre-commit hooks (one-time setup)

pre-commit install

# Run hooks manually on all files

pre-commit run --all-files

# Run hooks manually on staged files only

pre-commit run

# Skip hooks (use sparingly, only when necessary)

git commit --no-verify
```

**Configured Hooks**:

- `mojo format` - Auto-format Mojo code (`.mojo`, `.🔥` files)
- `markdownlint-cli2` - Lint markdown files (currently disabled, will enable after fixing existing files)
- `trailing-whitespace` - Remove trailing whitespace
- `end-of-file-fixer` - Ensure files end with newline
- `check-yaml` - Validate YAML syntax
- `check-added-large-files` - Prevent large files from being committed (max 1MB)
- `mixed-line-ending` - Fix mixed line endings

**CI Enforcement**: The `.github/workflows/pre-commit.yml` workflow runs these checks on all PRs and pushes to `main`.

## Repository Architecture

### Project Structure

```text
ml-odyssey/
├── agents/                      # Team documentation (tracked in git)
│   ├── README.md                # Quick start guide
│   ├── hierarchy.md             # Visual hierarchy diagram
│   ├── agent-hierarchy.md       # Complete agent specifications
│   ├── delegation-rules.md      # Coordination patterns
│   └── templates/               # Agent configuration templates
├── notes/
│   ├── plan/                    # 4-level hierarchical plans (LOCAL ONLY - not in git)
│   │   ├── 01-foundation/       # Repository structure and config
│   │   ├── 02-shared-library/   # Core reusable components
│   │   ├── 03-tooling/          # Development and testing tools
│   │   ├── 04-first-paper/      # LeNet-5 (proof of concept)
│   │   ├── 05-ci-cd/            # CI/CD pipelines
│   │   └── 06-agentic-workflows/# Claude-powered automation
│   ├── issues/                  # Issue-specific documentation (tracked in git)
│   │   ├── 62/README.md         # Issue #62: [Plan] Agents
│   │   ├── 63/README.md         # Issue #63: [Test] Agents
│   │   └── ...                  # One directory per issue
│   └── review/                  # Comprehensive specs & architectural decisions (tracked in git)
│       ├── agent-architecture-review.md
│       ├── skills-design.md
│       └── orchestration-patterns.md
├── scripts/                     # Python automation scripts
├── logs/                        # Execution logs and state files
└── .clinerules                 # Claude Code conventions
```

### Planning Hierarchy

**4 Levels** (in local `notes/plan/` directory):

1. **Section** (e.g., 01-foundation) - Major area of work
2. **Subsection** (e.g., 01-directory-structure) - Logical grouping
3. **Component** (e.g., 01-create-papers-dir) - Specific deliverable
4. **Subcomponent** (e.g., 01-create-base-dir) - Atomic task

Each component at any level has (locally, not in version control):

- `plan.md` - The planning document (Template 1 format, task-relative)
- `github_issue.md` - Dynamically generated GitHub issue definitions (task-relative)

**Important**: Plan files are for local planning and GitHub issue generation. Refer to tracked documentation in
`notes/issues/`, `notes/review/`, and `agents/` for shared team documentation.

### Documentation Organization

The repository uses three separate locations for documentation to avoid duplication:

#### 1. Team Documentation (`/agents/`)

**Purpose**: Quick start guides, visual references, and templates for team onboarding.

**Contents**:

- Quick start guides (README.md)
- Visual diagrams (hierarchy.md)
- Quick reference cards (delegation-rules.md)
- Configuration templates (templates/)

**When to Use**: Creating new documentation for team onboarding or quick reference.

#### 2. Comprehensive Specifications (`/notes/review/`)

**Purpose**: Detailed architectural decisions, comprehensive specifications, and design documents.

**Contents**:

- Architectural reviews and decisions
- Comprehensive design specifications (agent-hierarchy.md, skills-design.md)
- Workflow strategies (orchestration-patterns.md, worktree-strategy.md)
- Implementation summaries and lessons learned

**When to Use**: Writing detailed specifications, architectural decisions, or comprehensive guides.

#### 3. Issue-Specific Documentation (`/notes/issues/<issue-number>/`)

**Purpose**: Implementation notes, findings, and decisions specific to a single GitHub issue.

**Structure**: Each issue gets its own directory with a focused README.md:

```markdown

# Issue #XX: [Phase] Component Name

## Objective

What this specific issue accomplishes (1-2 sentences)

## Deliverables

- List of files/changes this issue creates

## Success Criteria

- Checklist of completion criteria

## References

- Links to shared documentation in /agents/ and /notes/review/
- NO duplication of comprehensive docs

## Implementation Notes

- Notes discovered during implementation
- Initially empty, filled as work progresses

```

**Important Rules**:

- ✅ DO: Link to comprehensive docs in `/agents/` and `/notes/review/`
- ✅ DO: Add issue-specific findings and decisions
- ❌ DON'T: Duplicate comprehensive documentation
- ❌ DON'T: Create shared specifications here (use `/notes/review/` instead)

### 5-Phase Development Workflow

Every component follows a hierarchical workflow with clear dependencies:

**Workflow**: Plan → [Test | Implementation | Packaging] → Cleanup

1. **Plan** - Design and documentation (MUST complete first)
2. **Test** - Write tests following TDD (parallel after Plan)
3. **Implementation** - Build the functionality (parallel after Plan)
4. **Packaging** - Integration and packaging (parallel after Plan)
5. **Cleanup** - Refactor and finalize (runs after parallel phases complete)

**Key Points**:

- Plan phase produces specifications for all other phases
- Test/Implementation/Packaging can run in parallel after Plan completes
- Cleanup collects issues discovered during the parallel phases
- Each phase has a separate GitHub issue with detailed instructions

## Plan File Format (Template 1)

**Note**: Plan files are task-relative and stored locally in `notes/plan/`.

All plan.md files follow this 9-section format:

```markdown

# Component Name

## Overview

Brief description (2-3 sentences)

## Parent Plan

[../plan.md](../plan.md) or "None (top-level)"

## Child Plans

- [child1/plan.md](child1/plan.md)

Or "None (leaf node)" for level 4

## Inputs

- Prerequisite 1

## Outputs

- Deliverable 1

## Steps

1. Step 1

## Success Criteria

- [ ] Criterion 1

## Notes

Additional context
```

**Important**: When modifying plans:

- Maintain all 9 sections consistently
- Use relative paths for links (e.g., `../plan.md`, not absolute paths)
- After editing plan.md, regenerate github_issue.md files using `scripts/regenerate_github_issues.py`
- NEVER edit github_issue.md files manually - they are dynamically generated
- **Remember**: Plan files are task-relative and NOT tracked in git - only reference tracked documentation

  (`notes/issues/`, `notes/review/`, `agents/`) in commits

## Script Architecture

### create_issues.py (854 LOC)

Main script for creating GitHub issues. Key features:

- Parses github_issue.md files and creates GitHub issues via `gh` CLI
- Automatic label creation with predefined colors
- Exponential backoff retry logic (up to 3 retries)
- State management with resume capability (saves every 10 issues)
- Updates markdown files with created issue URLs
- Comprehensive error handling and logging

### regenerate_github_issues.py (450+ LOC)

Generates github_issue.md files from plan.md sources:

- Extracts all sections from plan.md (overview, inputs, outputs, steps, criteria, notes)
- Generates consistent 5-issue format for each component
- Supports dry-run, section-by-section, and resume modes
- Timestamped state files for tracking multiple runs

### create_single_component_issues.py (198 LOC)

Testing utility for single component issue creation:

- Creates 5 issues for one component only
- Useful for validation before bulk creation
- Same label creation and markdown update logic as main script

## Working with Plans

**Important**: Plan files are task-relative and kept locally in `notes/plan/`. They are NOT tracked in version control.

### Creating a New Component

1. Create directory structure under `notes/plan/` (local only)
2. Create `plan.md` following Template 1 format (9 sections)
3. Update parent plan's "Child Plans" section
4. Regenerate github_issue.md: `python3 scripts/regenerate_github_issues.py --section <section>`
5. Test issue creation: `python3 scripts/create_single_component_issues.py notes/plan/.../github_issue.md`

### Modifying Existing Plans

1. Edit the `plan.md` file (maintain Template 1 format, local file)
2. Regenerate github_issue.md: `python3 scripts/regenerate_github_issues.py`
3. If issues were already created, update them manually in GitHub

### File Locations

- **Plans**: `notes/plan/<section>/<subsection>/.../plan.md` (LOCAL ONLY - not in git)
- **Generated Issues**: `notes/plan/<section>/<subsection>/.../github_issue.md` (LOCAL ONLY - DO NOT EDIT)
- **Scripts**: `scripts/*.py` (tracked in git)
- **Logs**: `logs/create_issues_*.log` (not tracked)
- **State**: `logs/.issue_creation_state_*.json` (not tracked)
- **Tracked Docs**: `notes/issues/<issue-number>/`, `notes/review/`, `agents/` (reference these in commits)

## Git Workflow

### Branch Naming

- `main` - Production branch
- `<issue-number>-<description>` - Feature/fix branches (e.g., `2-plan-create-base-directory`)

### Commit Message Format

Follow conventional commits:

```text
feat(section): Add new component
fix(scripts): Correct parsing issue
docs(readme): Update instructions
refactor(plans): Standardize to Template 1
```

## Labels

Standard labels automatically created by scripts:

- `planning` - Design phase (light purple: #d4c5f9)
- `documentation` - Documentation work (blue: #0075ca)
- `testing` - Testing phase (yellow: #fbca04)
- `tdd` - Test-driven development (yellow: #fbca04)
- `implementation` - Implementation phase (dark blue: #1d76db)
- `packaging` - Integration/packaging (light green: #c2e0c6)
- `integration` - Integration tasks (light green: #c2e0c6)
- `cleanup` - Cleanup/finalization (red: #d93f0b)

## Python Coding Standards

```python

#!/usr/bin/env python3

"""
Script description

Usage:
    python scripts/script_name.py [options]
"""

# Standard imports first

import sys
import re
from pathlib import Path
from typing import List, Dict, Optional

def function_name(param: str) -> bool:
    """Clear docstring with purpose, params, returns."""
    pass
```

**Requirements**:

- Python 3.7+
- Type hints required for all functions
- Clear docstrings for public functions
- Comprehensive error handling
- Logging for important operations

## Markdown Standards

All markdown files must follow these standards to pass `markdownlint-cli2` linting:

### Code Blocks (MD031, MD040)

**Rule**: Fenced code blocks must be:

1. Surrounded by blank lines (before and after)
2. Have a language specified

**Correct**:

```markdown

Some text before.

```python

def hello():
    print("world")

```

Some text after.

```text

**Incorrect**:

```markdown
Some text before.
```text

def hello():

```text
Some text after.
```

**Language Examples**:

- Python: ` ```python `
- Bash: ` ```bash `
- Text/plain: ` ```text `
- Mojo: ` ```mojo `
- YAML: ` ```yaml `
- JSON: ` ```json `
- Markdown: ` ```markdown `

### Lists (MD032)

**Rule**: Lists must be surrounded by blank lines (before and after)

**Correct**:

```markdown
Some text before.

- Item 1
- Item 2
- Item 3

Some text after.
```

**Incorrect**:

```markdown
Some text before.
- Item 1
- Item 2
Some text after.
```

### Headings (MD022)

**Rule**: Headings must be surrounded by blank lines (one blank line before and after)

**Correct**:

```markdown
Some content here.

## Section Heading

More content here.
```

**Incorrect**:

```markdown
Some content here.
## Section Heading
More content here.
```

### Line Length (MD013)

**Rule**: Lines should not exceed 120 characters (except for URLs or code blocks)

**Guidelines**:

- Break long lines at 120 characters
- For long sentences, break at natural boundaries (clauses, lists, etc.)
- Code in code blocks is exempt
- URLs in links are exempt (use reference-style links if needed)

**Example**:

```markdown
This is a very long sentence that exceeds the 120 character limit and should be broken into
multiple lines at a natural boundary point for better readability.
```

### Best Practices

1. **Always add blank lines around code blocks and lists** - This is the #1 cause of linting failures
2. **Always specify language for code blocks** - Use appropriate language tags
3. **Check headings have surrounding blank lines** - Especially after subheadings
4. **Use reference-style links for long URLs** - Helps avoid line length issues

### Quick Checklist for New Content

Before committing markdown files:

- [ ] All code blocks have a language specified (` ```python ` not ` ``` `)
- [ ] All code blocks have blank lines before and after
- [ ] All lists have blank lines before and after
- [ ] All headings have blank lines before and after
- [ ] No lines exceed 120 characters
- [ ] File ends with newline (enforced by pre-commit)
- [ ] No trailing whitespace (enforced by pre-commit)

### Running Markdown Linting Locally

```bash

# Check specific file
npx markdownlint-cli2 path/to/file.md

# Check all markdown files
pre-commit run markdownlint-cli2 --all-files

# View detailed errors
npx markdownlint-cli2 path/to/file.md 2>&1

```

## Debugging

### Check Logs

```bash

# View most recent log

tail -100 logs/create_issues_*.log | tail -100

# View specific log

cat logs/create_issues_20251107_180746.log
```

### Check State Files

```bash

# View saved state (for resume capability)

cat logs/.issue_creation_state_*.json
```

### Test Parsing

```bash

# Dry-run to test without creating issues

python3 scripts/create_issues.py --dry-run

# Test single component

python3 scripts/create_single_component_issues.py notes/plan/01-foundation/github_issue.md
```

## Troubleshooting

### GitHub CLI Issues

```bash

# Check authentication

gh auth status

# If missing scopes, refresh authentication

gh auth refresh -h github.com
```

### Issue Creation Failures

- Check GitHub CLI auth: `gh auth status`
- Verify repository access
- Check logs: `tail -100 logs/create_issues_*.log`
- Use `--resume` to continue from interruption

### Broken Links in Plans

- Use relative paths: `../plan.md` not absolute
- Verify files exist at referenced paths
- Update links if files are moved

### Script Errors

- Verify Python version: `python3 --version` (requires 3.7+)
- Check file permissions
- Review error logs in `logs/` directory

## Important Files

- `.clinerules` - Comprehensive Claude Code conventions (350+ lines)
- `notes/README.md` - GitHub issues creation plan
- `notes/review/README.md` - PR review guidelines and 5-phase workflow explanation
- `scripts/README.md` - Complete scripts documentation
- `README.md` - Main project documentation
