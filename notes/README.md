# GitHub Issues Creation Plan

## Overview

This repository uses a comprehensive 4-level hierarchical planning structure. Plan files are stored
locally in `notes/plan/` and are **NOT tracked in version control** - they are task-relative and used
for local planning and GitHub issue generation.

**Important**: For tracked team documentation, see `notes/issues/`, `notes/review/`, and `agents/` directories.

## Repository Structure

```text
ml-odyssey/
├── notes/
│   ├── plan/                    # task-relative planning
│   │   ├── 01-foundation/
│   │   ├── 02-shared-library/
│   │   ├── 03-tooling/
│   │   ├── 04-first-paper/
│   │   ├── 05-ci-cd/
│   │   └── 06-agentic-workflows/
│   ├── issues/                  # Tracked docs - historical issue documentation
│   │   └── 1/, 62-67/, etc.
│   └── review/                  # Tracked docs - PR review documentation
├── agents/                      # Tracked docs - agent system documentation
├── scripts/                     # Automation scripts
│   ├── create_issues.py         # Main issue creation script
│   ├── create_single_component_issues.py  # Test script
│   └── README.md                # Scripts documentation
└── logs/                        # Execution logs (not tracked)
```

## Structure

- **Sections**: 6 major areas
- **Levels**: 4-level hierarchy per section
- **GitHub Issues**: 5 issues per component (Plan, Test, Implementation, Packaging, Cleanup)

### Major Sections

| Section | Description |
|---------|-------------|
| 01-foundation | Repository structure and configuration |
| 02-shared-library | Core reusable components |
| 03-tooling | Development and testing tools |
| 04-first-paper | LeNet-5 implementation (proof of concept) |
| 05-ci-cd | Continuous integration and deployment |
| 06-agentic-workflows | Claude-powered automation |

### 5-Phase Workflow

Each component generates 5 issues following a hierarchical workflow:

**Workflow**: Plan → [Test | Implementation | Packaging] → Cleanup

1. **Plan** - Design and documentation (must complete first)
2. **Test** - Write tests following TDD (parallel after Plan)
3. **Implementation** - Build the functionality (parallel after Plan)
4. **Packaging** - Integration and packaging (parallel after Plan)
5. **Cleanup** - Refactor and finalize (runs after parallel phases complete)

## Quick Start

### Step 1: Test Issue Creation (✅ Complete)

We've verified issue creation works:

```bash

# Already tested - created 5 issues successfully
# Note: Plan files are local (notes/plan/)

python3 scripts/create_single_component_issues.py \
  notes/plan/01-foundation/01-directory-structure/01-create-papers-dir/01-create-base-dir/github_issue.md
```

Results: <https://github.com/mvillmow/ml-odyssey/issues/2-6>

### Step 2: Create All Issues

### Option A: Section by Section (Recommended)

```bash
# Create issues for one section at a time

python3 scripts/create_issues.py --section 01-foundation
python3 scripts/create_issues.py --section 02-shared-library
python3 scripts/create_issues.py --section 03-tooling
python3 scripts/create_issues.py --section 04-first-paper
python3 scripts/create_issues.py --section 05-ci-cd
python3 scripts/create_issues.py --section 06-agentic-workflows
```

### Option B: All at Once

```bash
# Create all issues

python3 scripts/create_issues.py
```

### Option C: Dry-Run First

```bash
# See what will be created without actually creating

python3 scripts/create_issues.py --dry-run
```

### Step 3: Resume if Interrupted

If the process is interrupted:

```bash
python3 scripts/create_issues.py --resume
```

## Features

### Automatic Label Creation

The scripts automatically create necessary labels:

- `planning` - Design and documentation phase
- `documentation` - Documentation tasks
- `testing` - Testing phase
- `tdd` - Test-driven development
- `implementation` - Implementation phase
- `packaging` - Integration and packaging
- `integration` - Integration tasks
- `cleanup` - Cleanup and finalization

### Progress Tracking

- Saves state every 10 issues
- Can resume from interruption
- Logs all actions to `logs/create_issues_TIMESTAMP.log`
- Updates github_issue.md files with issue URLs

### Error Handling

- Retries failed issues up to 3 times
- Exponential backoff between retries
- Continues processing even if some issues fail
- Comprehensive error logging

## File Updates

After issue creation, all github_issue.md files are automatically updated:

**Before:**

```markdown
**GitHub Issue URL**: [To be created]
```

**After:**

```markdown
**GitHub Issue URL**: https://github.com/mvillmow/ml-odyssey/issues/123
```

## Documentation

- **[scripts/README.md](../scripts/README.md)** - Complete scripts documentation
- **[notes/issues/](issues/)** - Historical issue documentation (tracked in git)
- **[notes/review/](review/)** - PR review documentation (tracked in git)
- **[agents/](../agents/)** - Agent system documentation (tracked in git)

**Note**: `notes/plan/` contains local planning files.
Reference the tracked documentation above for team collaboration.

## Workflow

### 5-Phase Development Process

Each component follows a 5-phase workflow:

1. **Plan** → Design and document the approach
2. **Test** → Write tests following TDD
3. **Implementation** → Build the functionality
4. **Packaging** → Integrate with existing code
5. **Cleanup** → Refactor and finalize

Each phase has its own GitHub issue with detailed instructions.

## Next Steps After Issue Creation

1. **Organize in GitHub Projects** - Group issues by section/milestone
2. **Create Milestones** - One per top-level section
3. **Begin Implementation** - Start with 01-foundation
4. **Create PRs** - One PR per component linking to its 5 issues

## Support

For questions or issues:

- Check the logs in `logs/`
- Review [scripts/README.md](../scripts/README.md)
- See existing issues on GitHub
- Refer to tracked documentation in `notes/issues/`, `notes/review/`, `agents/`

**Note**: Plan files in `notes/plan/` are task-relative (not in git).
Generate them locally as needed for issue creation.
