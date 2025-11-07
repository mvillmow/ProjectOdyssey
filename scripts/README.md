# Scripts Directory

This directory contains automation scripts for the Mojo AI Research Repository project.

## Overview

These scripts automate various aspects of the repository management, including:
- Creating GitHub issues from plan files
- Updating issue templates
- Managing repository structure

## Scripts

### 🎯 Main Scripts

#### `create_issues.py`
**Purpose**: Create GitHub issues from all github_issue.md files in the notes/plan directory.

**Features**:
- Parses 331 github_issue.md files
- Creates 1,655 GitHub issues (5 per component)
- Supports dry-run mode for testing
- Progress tracking and state management
- Automatic retry on failures
- Updates github_issue.md files with created issue URLs

**Usage**:
```bash
# Dry-run mode (shows what would be created without creating)
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

**Statistics**:
- Total files: 331
- Total issues: 1,655
- Sections: 6 (foundation, shared-library, tooling, first-paper, ci-cd, agentic-workflows)

**Output**:
- Logs to `logs/create_issues_TIMESTAMP.log`
- Saves state to `.issue_creation_state.json`
- Updates github_issue.md files with issue URLs

### 📝 Update Scripts

These scripts were used during the initial setup to populate github_issue.md files with detailed issue bodies.

#### `simple_update.py`
**Purpose**: Simple script for updating github_issue.md files with basic structure.

**Status**: Completed - used during initial setup phase

#### `update_agentic_workflows_issues.py`
**Purpose**: Update github_issue.md files in the 06-agentic-workflows section.

**Status**: Completed - all 52 files updated

#### `update_ci_cd_issues.py`
**Purpose**: Update github_issue.md files in the 05-ci-cd section.

**Status**: Completed - all 44 files updated

#### `update_tooling_issues.py`
**Purpose**: Update github_issue.md files in the 03-tooling section.

**Status**: Completed - all 53 files updated

**Note**: These update scripts were used to populate the github_issue.md files and are kept for reference. All 331 files have been updated with detailed issue bodies.

## Directory Structure

```
scripts/
├── README.md                              # This file
├── create_issues.py                       # Main issue creation script
├── simple_update.py                       # Basic update utility
├── update_agentic_workflows_issues.py     # Section-specific updater
├── update_ci_cd_issues.py                 # Section-specific updater
└── update_tooling_issues.py               # Section-specific updater
```

## Common Workflows

### Creating All GitHub Issues

**Step 1: Dry-run to verify**
```bash
cd /home/mvillmow/ml-odyssey
python3 scripts/create_issues.py --dry-run
```

Review the output to ensure all issues are parsed correctly.

**Step 2: Test with one section**
```bash
python3 scripts/create_issues.py --section 01-foundation
```

This creates 210 issues for the foundation section. Review them in GitHub to ensure they look correct.

**Step 3: Create all issues**
```bash
python3 scripts/create_issues.py
```

This creates all 1,655 issues. Takes approximately 30-45 minutes.

**Step 4: Verify completion**
```bash
# Check logs
tail -100 logs/create_issues_*.log

# Verify all files were updated
find notes/plan -name "github_issue.md" -exec grep -L "https://github.com" {} \;
```

An empty output means all files were updated successfully.

### Resuming After Interruption

If the script is interrupted:

```bash
python3 scripts/create_issues.py --resume
```

This resumes from the last saved state in `.issue_creation_state.json`.

## Requirements

### Python Packages
- Python 3.7+
- Standard library (no additional packages required for basic operation)
- Optional: `tqdm` for better progress bars
  ```bash
  pip install tqdm
  ```

### System Requirements
- GitHub CLI (`gh`) installed and authenticated
  ```bash
  # Install gh CLI
  # Ubuntu/Debian
  sudo apt install gh

  # Authenticate
  gh auth login
  ```

- Git repository with remote configured
  ```bash
  git remote -v
  # Should show: origin https://github.com/mvillmow/ml-odyssey.git
  ```

## Logging

All scripts log to the `logs/` directory:

```
logs/
├── create_issues_20251106_215918.log
├── create_issues_20251106_213006.log
└── ...
```

Log format:
```
2025-11-06 21:59:18,888 - INFO - Logging to /home/mvillmow/ml-odyssey/logs/create_issues_20251106_215918.log
2025-11-06 21:59:18,890 - INFO - Using repository: mvillmow/ml-odyssey
2025-11-06 21:59:19,100 - INFO - Found 331 github_issue.md files
2025-11-06 21:59:20,500 - INFO - Parsed 1655 issues
...
```

## State Management

The `create_issues.py` script saves progress to `.issue_creation_state.json`:

```json
{
  "last_processed_index": 100,
  "created_issues": [
    {
      "title": "[Plan] Create Base Directory - Design and Documentation",
      "issue_url": "https://github.com/mvillmow/ml-odyssey/issues/123",
      "file_path": "notes/plan/01-foundation/.../github_issue.md",
      "created": true
    }
  ],
  "timestamp": "2025-11-06T21:59:18"
}
```

This allows the script to resume if interrupted.

## Error Handling

All scripts include robust error handling:

1. **Retry Logic**: Failed operations are retried up to 3 times
2. **Exponential Backoff**: Waits 1s, 2s, 4s between retries
3. **State Saving**: Progress saved every 10 issues
4. **Error Logging**: All errors logged with full traceback
5. **Graceful Degradation**: Continues processing even if individual issues fail

## Troubleshooting

### Issue: Script can't find git repository
**Solution**: Run from repository root:
```bash
cd /home/mvillmow/ml-odyssey
python3 scripts/create_issues.py --dry-run
```

### Issue: GitHub API rate limit
**Solution**: The script automatically handles rate limits with exponential backoff. Wait for it to retry.

### Issue: Permission denied
**Solution**: Ensure GitHub CLI is authenticated:
```bash
gh auth status
gh auth login  # If not authenticated
```

### Issue: Script interrupted
**Solution**: Resume from saved state:
```bash
python3 scripts/create_issues.py --resume
```

### Issue: Parse errors in github_issue.md files
**Solution**: Check the log file for details:
```bash
tail -100 logs/create_issues_*.log | grep ERROR
```

## Development

### Adding New Scripts

When adding new scripts to this directory:

1. **Add executable permission**:
   ```bash
   chmod +x scripts/your_script.py
   ```

2. **Add shebang line** at the top:
   ```python
   #!/usr/bin/env python3
   ```

3. **Include docstring** with description and usage:
   ```python
   """
   Script Name

   Description of what the script does.

   Usage:
       python scripts/your_script.py [options]
   """
   ```

4. **Update this README** with script documentation

### Testing Scripts

Before running scripts on production data:

1. **Test with dry-run** (if available)
2. **Test on a small subset** (e.g., one section)
3. **Review logs** for errors
4. **Verify output** manually

## Related Documentation

- [notes/README.md](../notes/README.md) - Detailed plan for issue creation
- [SCRIPTS_ANALYSIS.md](SCRIPTS_ANALYSIS.md) - Comprehensive scripts analysis and documentation
- [notes/plan/](../notes/plan/) - All plan files and issue templates
- [README.md](../README.md) - Main repository README

## Support

For issues or questions:
1. Check the logs in `logs/`
2. Review this README
3. Check the main project documentation
4. Create an issue in the repository

## License

Same as the main repository (Apache 2.0).
