# Scripts Directory

This directory contains Python automation scripts for the Mojo AI Research Repository project.

## Overview

These scripts automate repository management tasks:

- Creating GitHub issues from plan files (local files in `notes/plan/`, not tracked in git)
- Regenerating github_issue.md files dynamically from plan.md files (local, task-relative)
- Testing issue creation for individual components

**Important**: Plan files in `notes/plan/` are task-relative and NOT tracked in version control.
They are used for local planning and GitHub issue generation. For tracked team documentation,
see `notes/issues/`, `notes/review/`, and `agents/`.

## Scripts

### Main Scripts

#### `regenerate_github_issues.py`

**Purpose**: Regenerate all github_issue.md files dynamically from their corresponding plan.md files
(local files in `notes/plan/`, not tracked in git).

**Features**:

- Generates github_issue.md files from plan.md sources (task-relative, local only)
- Supports dry-run mode for testing
- Section-by-section processing
- Resume capability with timestamped state files
- Progress tracking and error handling

**Usage**:

```bash

# Dry-run to preview changes

python3 scripts/regenerate_github_issues.py --dry-run

# Regenerate one section

python3 scripts/regenerate_github_issues.py --section 01-foundation

# Regenerate all files

python3 scripts/regenerate_github_issues.py

# Resume from previous run

python3 scripts/regenerate_github_issues.py --resume
```

**Command-line Options**:

- `--dry-run`: Show what would be done without making changes
- `--section SECTION`: Process only one section (e.g., 01-foundation)
- `--resume`: Resume from last saved state
- `--plan-dir PATH`: Specify plan directory (default: /home/mvillmow/ml-odyssey/notes/plan)

**Output**:

- Logs to stderr with progress updates
- Saves state to `logs/.issue_creation_state_<timestamp>.json`
- Updates github_issue.md files in place

**Important**: github_issue.md files are dynamically generated and should not be edited manually.
Always regenerate them using this script. These files are local (in `notes/plan/`) and NOT tracked in git.

---

#### `create_issues.py`

**Purpose**: Create GitHub issues from all github_issue.md files in the notes/plan directory
(local files, not tracked in git).

**Features**:

- Creates GitHub issues using the gh CLI
- Supports dry-run mode for testing
- Section-by-section processing
- Progress tracking and state management
- Automatic retry on failures with exponential backoff
- Updates github_issue.md files with created issue URLs
- Automatic label creation

**Usage**:

```bash

# Dry-run mode (recommended first - shows what would be created)

python3 scripts/create_issues.py --dry-run

# Test with one section only

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

**Command-line Options**:

- `--dry-run`: Show what would be done without creating issues
- `--section SECTION`: Process only one section (e.g., 01-foundation)
- `--resume`: Resume from last saved state
- `--no-color`: Disable ANSI color output
- `--repo REPO`: Override repository (default: auto-detected from git)

**Output**:

- Logs to `logs/create_issues_<timestamp>.log`
- Saves state to `logs/.issue_creation_state_<timestamp>.json`
- Updates github_issue.md files with issue URLs

**Prerequisites**:

- GitHub CLI (`gh`) must be installed and authenticated
- Run `gh auth login` if not already authenticated

---

#### `create_single_component_issues.py`

**Purpose**: Test script to create GitHub issues for a single component.

**Features**:

- Creates 5 issues for one component (Plan, Test, Implementation, Packaging, Cleanup)
- Useful for testing before bulk creation
- Updates github_issue.md with created issue URLs

**Usage**:

```bash

# Test with a specific component

python3 scripts/create_single_component_issues.py notes/plan/01-foundation/github_issue.md
```

**Example**:

```bash

# This creates 5 issues for the foundation component

python3 scripts/create_single_component_issues.py notes/plan/01-foundation/github_issue.md
```

---

## Workflow

### Typical Development Flow

**Note**: Plan files are local (in `notes/plan/`) and NOT tracked in git.

1. **Edit plan.md files** - Make changes to local planning documents
2. **Regenerate github_issue.md** - Run `regenerate_github_issues.py` to update local issue files
3. **Test with dry-run** - Run `create_issues.py --dry-run` to preview
4. **Create issues** - Run `create_issues.py` to create GitHub issues from local plans

### Creating All Issues

```bash

# Step 1: Ensure github_issue.md files are up to date

python3 scripts/regenerate_github_issues.py

# Step 2: Dry-run to verify

python3 scripts/create_issues.py --dry-run

# Step 3: Create issues section-by-section (recommended)

python3 scripts/create_issues.py --section 01-foundation
python3 scripts/create_issues.py --section 02-shared-library
python3 scripts/create_issues.py --section 03-tooling
python3 scripts/create_issues.py --section 04-first-paper
python3 scripts/create_issues.py --section 05-ci-cd
python3 scripts/create_issues.py --section 06-agentic-workflows

# OR: Create all at once

python3 scripts/create_issues.py
```

---

## File Locations

### State Files

State files are saved with timestamps in the `logs/` directory:

- `logs/.issue_creation_state_<timestamp>.json`
- Contains processed files list and completion status
- Used for resume capability

### Log Files

Execution logs are saved in the `logs/` directory:

- `logs/create_issues_<timestamp>.log`
- Contains detailed progress and error information

### GitHub Issue Files

github_issue.md files are dynamically generated (local, NOT tracked in git):

- Located in `notes/plan/**/github_issue.md` (task-relative, not in version control)
- Generated from corresponding plan.md files (also local, not tracked)
- Regenerated locally as needed for issue creation
- Each contains 5 issue definitions (Plan, Test, Implementation, Packaging, Cleanup)

---

## Issue Structure

Each component generates 5 GitHub issues following the 5-phase development workflow:

### 5-Phase Hierarchy

**Workflow**: Plan → [Test | Implementation | Packaging] → Cleanup

1. **Plan Issue** - `[Plan] Component Name - Design and Documentation`
   - Labels: `planning`, `documentation`
   - Purpose: Create detailed specifications and design
   - Must complete before other phases

2. **Test Issue** - `[Test] Component Name - Write Tests`
   - Labels: `testing`, `tdd`
   - Purpose: Document and implement test cases
   - Can run in parallel with Implementation and Packaging

3. **Implementation Issue** - `[Impl] Component Name - Implementation`
   - Labels: `implementation`
   - Purpose: Build the main functionality
   - Can run in parallel with Test and Packaging

4. **Packaging Issue** - `[Package] Component Name - Integration and Packaging`
   - Labels: `packaging`, `integration`
   - Purpose: Integrate artifacts and create installer
   - Can run in parallel with Test and Implementation

5. **Cleanup Issue** - `[Cleanup] Component Name - Refactor and Finalize`
   - Labels: `cleanup`, `documentation`
   - Purpose: Collect issues, refactor, and finalize
   - Runs after parallel phases complete

See [notes/review/README.md](../notes/review/README.md) for detailed workflow documentation.

---

## Troubleshooting

### GitHub CLI Not Installed

```bash

# Install GitHub CLI
# See: https://github.com/cli/cli#installation

# Authenticate

gh auth login
```

### State File Issues

If you need to start fresh:

```bash

# State files are in logs/ with timestamps
# Remove specific state file or let script create new one

rm logs/.issue_creation_state_*.json
```

### Permission Errors

Ensure you have:

- Write access to `logs/` directory
- Write access to `notes/plan/` directory (local files, for updating github_issue.md)
- GitHub repository write access

**Note**: `notes/plan/` is local and NOT tracked in git.

### Rate Limiting

GitHub API has rate limits. If you hit them:

- Wait for rate limit to reset
- Use `--resume` to continue from where you left off
- Process sections one at a time with delays between them

### Missing github_issue.md Files

If github_issue.md files are missing:

```bash

# Regenerate all files

python3 scripts/regenerate_github_issues.py
```

---

## Script Dependencies

### Python Requirements

- Python 3.7+
- No external Python packages required (uses standard library only)

### External Tools

- `gh` (GitHub CLI) - Required for creating issues
- `git` - Required for repository detection

### Repository Structure

Scripts expect this structure:

```text
ml-odyssey/
├── notes/
│   ├── plan/                # LOCAL ONLY (not in git) - task-relative planning
│   │   ├── 01-foundation/
│   │   │   ├── plan.md
│   │   │   └── (github_issue.md - generated)
│   │   └── ...
│   ├── issues/              # Tracked docs - historical issue documentation
│   └── review/              # Tracked docs - PR review documentation
├── agents/                  # Tracked docs - agent system documentation
├── scripts/
│   ├── create_issues.py
│   ├── create_single_component_issues.py
│   └── regenerate_github_issues.py
└── logs/                    # Not tracked
    ├── .issue_creation_state_*.json
    └── create_issues_*.log
```

---

## Best Practices

1. **Always dry-run first**
   - Use `--dry-run` to preview changes before executing
   - Verify output looks correct

2. **Test with one component**
   - Use `create_single_component_issues.py` to test with one component
   - Verify issues are created correctly

3. **Process section-by-section**
   - Use `--section` flag for better control
   - Easier to handle errors and rate limits

4. **Check logs**
   - Review log files in `logs/` directory
   - Contains detailed error messages and progress

5. **Use resume capability**
   - If interrupted, use `--resume` to continue
   - State is saved every 50 files

6. **Keep github_issue.md files in sync**
   - Regenerate after editing plan.md files
   - Don't edit github_issue.md manually
   - Remember: These are local files (not tracked in git)

---

## Script Details

### create_issues.py

- **Lines of Code**: 854
- **Complexity**: High (comprehensive error handling, state management, retry logic)
- **Key Features**:
  - Automatic label creation with predefined colors
  - Exponential backoff retry logic (up to 3 retries)
  - Progress tracking with optional tqdm support
  - Colored terminal output (can be disabled)
  - State saved every 10 issues for resume capability
  - Parses multiple github_issue.md format variations
  - Updates markdown files with created issue URLs

**Label Colors**:

```python
'planning': 'd4c5f9'       # Light purple
'documentation': '0075ca'  # Blue
'testing': 'fbca04'        # Yellow
'tdd': 'fbca04'           # Yellow
'implementation': '1d76db' # Dark blue
'packaging': 'c2e0c6'      # Light green
'integration': 'c2e0c6'    # Light green
'cleanup': 'd93f0b'        # Red
```

### create_single_component_issues.py

- **Lines of Code**: 198
- **Complexity**: Medium
- **Purpose**: Testing and verification tool for single components
- **Features**:
  - Same label creation as main script
  - Simpler focused implementation for testing
  - Direct markdown file updates
  - Useful for validating changes before bulk creation

### regenerate_github_issues.py

- **Lines of Code**: 450+
- **Complexity**: Medium
- **Purpose**: Dynamic generation of github_issue.md files from plan.md sources
- **Features**:
  - Extracts all sections from plan.md (overview, inputs, outputs, steps, criteria, notes)
  - Generates consistent 5-issue format for each component
  - Supports dry-run, section-by-section, and resume modes
  - Timestamped state files for tracking multiple runs
  - Replaces 4 legacy update scripts with single consolidated tool

**Body Generation**:

- **Plan**: Objectives, inputs, outputs, success criteria, notes
- **Test**: Testing objectives, what to test, test success criteria, implementation steps
- **Implementation**: Goals, required inputs, outputs, implementation steps, success criteria
- **Packaging**: Objectives, integration requirements, integration steps, success criteria
- **Cleanup**: Objectives, cleanup tasks, success criteria, notes

---

## Related Documentation

- [Repository README](../README.md) - Main project documentation
- [Planning Documentation](../notes/README.md) - GitHub issues plan overview
- [Review Process](../notes/review/README.md) - PR review guidelines and 5-phase workflow
- [Project Conventions](../.clinerules) - Claude Code conventions

---

## Notes

- All scripts use Python 3 standard library only
- github_issue.md files are dynamically generated, not committed (local only)
- plan.md files are task-relative and NOT tracked in git
- State files include timestamps for tracking multiple runs
- Scripts handle errors gracefully with detailed logging
- Resume capability prevents duplicate work if interrupted
- For tracked team documentation, see `notes/issues/`, `notes/review/`, and `agents/`
