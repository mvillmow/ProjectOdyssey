# Scripts Directory

This directory contains Python automation scripts for the Mojo AI Research Repository project.

## Overview

These scripts automate repository management tasks:

- Creating GitHub issues from plan files (files in `notes/plan/`)
- Regenerating github_issue.md files dynamically from plan.md files (local, task-relative)
- Testing issue creation for individual components
- Agent system utilities and validation

**Important**: Plan files in `notes/plan/` are task-relative.
They are used for local planning and GitHub issue generation. For tracked team documentation,
see `notes/issues/`, `notes/review/`, and `agents/`.

## Directory Structure

```text
scripts/
├── README.md                           # This file
├── common.py                           # Shared utilities and constants
├── create_issues.py                    # Main GitHub issue creation
├── create_single_component_issues.py   # Single component testing
├── regenerate_github_issues.py         # Dynamic issue file generation
├── fix_markdown.py                     # Unified markdown linting fixer
├── validate_links.py                   # Markdown link validation
├── validate_structure.py               # Repository structure validation
├── check_readmes.py                    # README completeness validation
├── lint_configs.py                     # YAML configuration linting
├── get_system_info.py                  # System information collector
├── merge_prs.py                        # PR merge automation
├── package_papers.py                   # Papers directory packaging
├── fix_duplicate_delegation.py         # Agent refactoring (legacy)
├── cleanup_agent_redundancy.py         # Agent refactoring (legacy)
├── fix_agent_markdown.py               # Agent refactoring (legacy)
├── condense_pr_sections.py             # Agent refactoring (legacy)
├── condense_mojo_guidelines.py         # Agent refactoring (legacy)
└── agents/                             # Agent system utilities
    ├── README.md                       # Agent scripts documentation
    ├── agent_health_check.sh           # System health checks
    ├── agent_stats.py                  # Statistics and metrics
    ├── check_frontmatter.py            # YAML validation
    ├── list_agents.py                  # Agent discovery
    ├── setup_agents.sh                 # Setup automation
    ├── test_agent_loading.py           # Loading tests
    ├── validate_agents.py              # Configuration validation
    └── tests/                          # Agent test suite
```

## Scripts

### Main Scripts

#### `regenerate_github_issues.py`

**Purpose**: Regenerate all github_issue.md files dynamically from their corresponding plan.md files.

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
- `--plan-dir PATH`: Specify plan directory (default: auto-detected from repository root)

**Output**:

- Logs to stderr with progress updates
- Saves state to `logs/.issue_creation_state_<timestamp>.json`
- Updates github_issue.md files in place

**Important**: github_issue.md files are dynamically generated and should not be edited manually.
Always regenerate them using this script. These files are local (in `notes/plan/`) and NOT tracked in git.

---

#### `create_issues.py`

**Purpose**: Create GitHub issues from all github_issue.md files in the notes/plan directory

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

#### `fix_markdown.py`

**Purpose**: Unified markdown linting fixer that automatically fixes common markdownlint-cli2 errors.

**Features**:

- Fixes 8 common markdown linting rules (MD012, MD022, MD026, MD029, MD031, MD032, MD036, MD040)
- Supports single files or entire directories
- Dry-run mode to preview changes
- Verbose output option
- Excludes common directories (node_modules, .git, venv, etc.)

**Usage**:

```bash
# Fix a single file
python3 scripts/fix_markdown.py README.md

# Fix all markdown in a directory
python3 scripts/fix_markdown.py notes/

# Fix all markdown in repository
python3 scripts/fix_markdown.py .

# Dry run (preview without changes)
python3 scripts/fix_markdown.py . --dry-run

# Verbose output
python3 scripts/fix_markdown.py . --verbose
```

**Fixes Applied**:

- **MD012**: Remove multiple consecutive blank lines
- **MD022**: Add blank lines around headings
- **MD026**: Remove trailing punctuation from headings
- **MD029**: Fix ordered list numbering (use 1. for all items)
- **MD031**: Add blank lines around code blocks
- **MD032**: Add blank lines around lists
- **MD036**: Convert bold text used as headings to actual headings
- **MD040**: Add language tags to code blocks (defaults to `text`)

**Command-line Options**:

- `path`: Path to markdown file or directory (required)
- `-v, --verbose`: Enable verbose output
- `-n, --dry-run`: Show what would be fixed without making changes

**Example Output**:

```text
Found 42 markdown file(s)
Fixed notes/issues/3/README.md: 5 issues
Fixed scripts/README.md: 3 issues

Summary:
  Files modified: 2
  Total fixes: 8
```

---

#### `validate_links.py`

**Purpose**: Validate all markdown links in the repository.

**Features**:

- Checks relative file links and anchors
- Validates internal markdown links
- Reports broken links with line numbers
- Supports dry-run mode
- Exit code indicates validation status

**Usage**:

```bash
# Validate links in all markdown files
python3 scripts/validate_links.py

# Validate links in specific directory
python3 scripts/validate_links.py notes/
```

**Exit Codes**:

- 0 = All links valid
- 1 = Broken links found

---

#### `validate_structure.py`

**Purpose**: Validate repository directory structure and required files.

**Features**:

- Checks for required directories
- Validates presence of key files
- Section-by-section validation
- Clear error reporting

**Usage**:

```bash
# Validate repository structure
python3 scripts/validate_structure.py

# Validate specific section
python3 scripts/validate_structure.py --section 01-foundation
```

**Exit Codes**:

- 0 = Structure valid
- 1 = Validation errors found

---

#### `check_readmes.py`

**Purpose**: Validate README.md files for completeness and consistency.

**Features**:

- Checks required sections in README files
- Validates markdown formatting
- Reports missing sections
- Comprehensive validation

**Usage**:

```bash
# Check all README files
python3 scripts/check_readmes.py

# Check specific directory
python3 scripts/check_readmes.py notes/
```

**Exit Codes**:

- 0 = All README files valid
- 1 = Issues found

---

#### `lint_configs.py`

**Purpose**: Lint YAML configuration files for syntax and formatting.

**Features**:

- YAML syntax validation
- Indentation checking
- Nested key validation
- Can remove unused sections
- Verbose output mode

**Usage**:

```bash
# Lint all YAML files
python3 scripts/lint_configs.py

# Lint with verbose output
python3 scripts/lint_configs.py --verbose

# Remove unused sections
python3 scripts/lint_configs.py --remove-unused
```

**Exit Codes**:

- 0 = All configs valid
- 1 = Linting errors found

---

#### `get_system_info.py`

**Purpose**: Collect system information for bug reports and debugging.

**Features**:

- Gathers git information
- Collects Mojo version details
- Python environment info
- System details (OS, CPU, etc.)
- Graceful handling of missing tools

**Usage**:

```bash
# Collect system information
python3 scripts/get_system_info.py

# Output in JSON format
python3 scripts/get_system_info.py --json
```

---

#### `merge_prs.py`

**Purpose**: Merge pull requests using GitHub API.

**Features**:

- Uses PyGithub library
- Configurable merge method
- Delete branch after merge option
- Requires GITHUB_TOKEN

**Usage**:

```bash
# Merge PR by number
python3 scripts/merge_prs.py <pr-number>

# Merge with squash
python3 scripts/merge_prs.py <pr-number> --squash

# Delete branch after merge
python3 scripts/merge_prs.py <pr-number> --delete-branch
```

**Requirements**:

- PyGithub library: `pip install PyGithub`
- GITHUB_TOKEN environment variable

---

#### `package_papers.py`

**Purpose**: Create tarball distribution of papers directory.

**Features**:

- Creates compressed archive
- Validates directory structure
- Timestamped output files
- Error handling

**Usage**:

```bash
# Create papers package
python3 scripts/package_papers.py

# Specify output directory
python3 scripts/package_papers.py --output dist/
```

---

### Agent Modification Scripts

These scripts were used to refactor agent configuration files. They are kept for reference but may not be needed for regular development.

#### `fix_duplicate_delegation.py`

**Purpose**: Fix duplicate Delegation sections in agent files.

**Usage**:

```bash
python3 scripts/fix_duplicate_delegation.py
```

#### `cleanup_agent_redundancy.py`

**Purpose**: Remove redundant sections from agent files and replace with references.

**Usage**:

```bash
python3 scripts/cleanup_agent_redundancy.py
```

#### `fix_agent_markdown.py`

**Purpose**: Fix markdown linting errors in agent configuration files.

**Usage**:

```bash
python3 scripts/fix_agent_markdown.py
```

#### `condense_pr_sections.py`

**Purpose**: Replace verbose PR creation sections with concise references.

**Usage**:

```bash
python3 scripts/condense_pr_sections.py
```

#### `condense_mojo_guidelines.py`

**Purpose**: Condense Mojo-specific guidelines sections in agent files.

**Usage**:

```bash
python3 scripts/condense_mojo_guidelines.py
```

**Note**: These agent modification scripts use shared utilities from `common.py` for path detection and may be archived in the future if no longer needed.

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
- Generated from corresponding plan.md files
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
│   ├── issues/              # Tracked docs - issue-specific documentation
│   └── review/              # Tracked docs - PR review documentation
├── agents/                  # Tracked docs - agent system documentation
├── scripts/
│   ├── create_issues.py
│   ├── create_single_component_issues.py
│   ├── regenerate_github_issues.py
│   ├── agents/              # Agent utilities
│   └── archive/             # Historical scripts
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
