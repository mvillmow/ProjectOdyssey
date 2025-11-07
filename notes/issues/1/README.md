# GitHub Issue #1 - Create the Plan for the Mojo AI Research Repository

**Issue URL**: https://github.com/mvillmow/ml-odyssey/issues/1

This directory contains historical documentation from the initial planning work completed for GitHub Issue #1.

## Overview

GitHub Issue #1 established the complete planning foundation for the ML Odyssey repository, creating a comprehensive 4-level hierarchical structure across multiple major sections (foundation, shared library, tooling, first paper implementation, CI/CD, and agentic workflows).

## What Was Accomplished

### Planning Structure
- Created complete 4-level hierarchical planning structure
- All components have corresponding `plan.md` files following Template 1 format
- Proper parent-child linking throughout the hierarchy
- All plans include: Overview, Inputs, Outputs, Steps, Success Criteria, and Notes sections

### GitHub Issues Planning
- Designed 5-phase development workflow (Plan → [Test | Implementation | Packaging] → Cleanup)
- Created system for generating GitHub issues dynamically from plan.md files
- Each component has 5 associated issues following the workflow

### Automation Scripts
- **create_issues.py**: Main script to create GitHub issues from plan files
- **create_single_component_issues.py**: Testing utility for single components
- **regenerate_github_issues.py**: Consolidated script to regenerate github_issue.md files from plan.md files

## Key Scripts

### Regenerating GitHub Issue Files

Since github_issue.md files are dynamically generated, use the regeneration script when needed:

```bash
# Regenerate all github_issue.md files
python3 scripts/regenerate_github_issues.py

# Dry-run to preview changes
python3 scripts/regenerate_github_issues.py --dry-run

# Regenerate only one section
python3 scripts/regenerate_github_issues.py --section 01-foundation

# Resume from previous run
python3 scripts/regenerate_github_issues.py --resume
```

### Creating GitHub Issues

To create the actual GitHub issues:

```bash
# Dry-run mode (recommended first - shows what would be created)
python3 scripts/create_issues.py --dry-run

# Process only one section
python3 scripts/create_issues.py --section 01-foundation

# Resume from interruption
python3 scripts/create_issues.py --resume

# Create all issues
python3 scripts/create_issues.py

# Disable colored output
python3 scripts/create_issues.py --no-color

# Specify repository explicitly
python3 scripts/create_issues.py --repo username/repo
```

## 5-Phase Development Workflow

Each component follows a hierarchical development process:

**Workflow**: Plan → [Test | Implementation | Packaging] → Cleanup

1. **Plan** (Sequential - must complete first)
   - Creates detailed specifications
   - Defines requirements for other phases
   - Documents design decisions

2. **Test** (Parallel after Plan)
   - Documents and implements test cases
   - Follows TDD principles
   - Can run in parallel with Implementation and Packaging

3. **Implementation** (Parallel after Plan)
   - Main implementation work
   - Can run in parallel with Test and Packaging

4. **Packaging** (Parallel after Plan)
   - Integrates Test and Implementation artifacts
   - Creates reproducible installer/package
   - Can run in parallel with Test and Implementation

5. **Cleanup** (Sequential - runs after parallel phases)
   - Collects issues discovered during other phases
   - Handles refactoring and finalization
   - Final validation

See [notes/review/README.md](../review/README.md) for detailed workflow documentation.

## Issue Generation Format

Each component's github_issue.md file contains 5 issues with the following structure:

- **Plan Issue**: `[Plan] Component Name - Design and Documentation`
  - Labels: planning, documentation
  - Includes: Overview, Planning Tasks, Deliverables, Reference to plan.md

- **Test Issue**: `[Test] Component Name - Write Tests`
  - Labels: testing, tdd
  - Includes: Testing Objectives, Test Tasks, Test Types, Acceptance Criteria

- **Implementation Issue**: `[Impl] Component Name - Implementation`
  - Labels: implementation
  - Includes: Implementation Goals, Required Inputs, Implementation Steps, Success Criteria

- **Packaging Issue**: `[Package] Component Name - Integration and Packaging`
  - Labels: packaging, integration
  - Includes: Packaging Objectives, Integration Requirements, Integration Steps, Validation

- **Cleanup Issue**: `[Cleanup] Component Name - Refactor and Finalize`
  - Labels: cleanup, documentation
  - Includes: Cleanup Objectives, Cleanup Tasks, Documentation Tasks, Final Validation

## File Organization

### Plan Structure
```
notes/plan/
├── 01-foundation/
├── 02-shared-library/
├── 03-tooling/
├── 04-first-paper/
├── 05-ci-cd/
└── 06-agentic-workflows/
```

Each directory contains:
- `plan.md` - Component plan following Template 1 format
- (github_issue.md files are generated dynamically, not committed to repository)

### Scripts
```
scripts/
├── create_issues.py                 # Main issue creation script
├── create_single_component_issues.py  # Test single component
├── regenerate_github_issues.py      # Regenerate github_issue.md files
├── README.md                        # Scripts documentation
└── SCRIPTS_ANALYSIS.md              # Comprehensive scripts analysis
```

### Logs
```
logs/
├── create_issues_*.log                   # Issue creation logs
└── .issue_creation_state_*.json          # State files with timestamps
```

## Important Notes

### GitHub Issue Files
- github_issue.md files are **dynamically generated** from plan.md files
- They are NOT committed to the repository
- Use `scripts/regenerate_github_issues.py` to generate them when needed
- Never edit github_issue.md files manually - changes will be overwritten

### State Management
- Issue creation progress is saved to timestamped state files in `logs/`
- State files: `logs/.issue_creation_state_<timestamp>.json`
- Use `--resume` flag to continue from previous run

### Script Features
- **Dry-run mode**: Preview changes without creating issues
- **Section-by-section**: Process one section at a time
- **Resume capability**: Continue from interruption
- **Progress tracking**: Detailed logging and state management
- **Error handling**: Automatic retry with exponential backoff

## Troubleshooting

### Missing github_issue.md Files
If github_issue.md files are missing, regenerate them:
```bash
python3 scripts/regenerate_github_issues.py
```

### Script Failures
- Check `logs/create_issues_*.log` for detailed error messages
- Use `--dry-run` to test before making changes
- Ensure you have GitHub CLI (gh) installed and authenticated
- Verify write access to logs/ directory

### Resume from Failure
If script is interrupted:
```bash
python3 scripts/create_issues.py --resume
```

## Related Documentation

- [Repository README](../../README.md) - Main project documentation
- [Planning Documentation](../README.md) - GitHub issues plan overview
- [Scripts Documentation](../../scripts/README.md) - Detailed scripts reference
- [Review Process](../review/README.md) - PR review guidelines and 5-phase workflow
- [Project Conventions](.clinerules) - Claude Code conventions

## Historical Files

This directory contains the following historical documentation files from the initial planning work:
- IMPLEMENTATION_STATUS.md - Status report from initial implementation
- UPDATE_INSTRUCTIONS.md - Original update instructions
- UPDATE_SUMMARY.md - Summary of updates performed
- GITHUB_ISSUES_PLAN.md - Original issues creation plan

These files are preserved for historical context but may contain outdated information. Refer to the scripts and current documentation for up-to-date instructions.
