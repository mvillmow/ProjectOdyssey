# GitHub Issues Creation Plan

## Overview

The `create_issues.py` script will create **1,655 GitHub issues** from 331 github_issue.md files across 6 top-level sections.

## Statistics

### Total Breakdown
- **Files to process**: 331 github_issue.md files
- **Issues to create**: 1,655 (5 per file)
- **Repository**: mvillmow/ml-odyssey

### By Section
| Section | Components | Issues |
|---------|-----------|--------|
| 01-foundation | 42 | 210 |
| 02-shared-library | 57 | 285 |
| 03-tooling | 53 | 265 |
| 04-first-paper | 83 | 415 |
| 05-ci-cd | 44 | 220 |
| 06-agentic-workflows | 52 | 260 |
| **TOTAL** | **331** | **1,655** |

### By Type
| Issue Type | Count |
|------------|-------|
| Plan | 331 |
| Test | 331 |
| Implementation | 331 |
| Packaging | 331 |
| Cleanup | 331 |
| **TOTAL** | **1,655** |

## Example Issues

### Issue 1: [Plan] Create Base Directory - Design and Documentation
- **File**: `notes/plan/01-foundation/01-directory-structure/01-create-papers-dir/01-create-base-dir/github_issue.md`
- **Labels**: `planning`, `documentation`
- **Body**: 36 lines
- **Command**:
  ```bash
  gh issue create \
    --title "[Plan] Create Base Directory - Design and Documentation" \
    --body-file /tmp/issue_body_xxx.md \
    --label planning,documentation \
    --repo mvillmow/ml-odyssey
  ```

### Issue 2: [Test] Create Base Directory - Write Tests
- **File**: Same as above
- **Labels**: `testing`, `tdd`
- **Body**: 49 lines
- **Command**:
  ```bash
  gh issue create \
    --title "[Test] Create Base Directory - Write Tests" \
    --body-file /tmp/issue_body_xxx.md \
    --label testing,tdd \
    --repo mvillmow/ml-odyssey
  ```

### Issue 3: [Impl] Create Base Directory - Implementation
- **File**: Same as above
- **Labels**: `implementation`
- **Body**: 51 lines

### Issue 4: [Package] Create Base Directory - Integration and Packaging
- **File**: Same as above
- **Labels**: `packaging`, `integration`
- **Body**: 53 lines

### Issue 5: [Cleanup] Create Base Directory - Refactor and Finalize
- **File**: Same as above
- **Labels**: `cleanup`, `documentation`
- **Body**: 63 lines

## Script Features

### Command-Line Options
```bash
# Dry-run mode (recommended first)
python3 scripts/create_issues.py --dry-run

# Process only one section
python3 scripts/create_issues.py --section 01-foundation --dry-run
python3 scripts/create_issues.py --section 02-shared-library

# Resume from interruption
python3 scripts/create_issues.py --resume

# Actually create all issues
python3 scripts/create_issues.py

# Disable colors
python3 scripts/create_issues.py --no-color

# Override repository
python3 scripts/create_issues.py --repo username/repo
```

### Features
1. **Dry-run mode**: Shows what will be created without actually creating
2. **Progress tracking**: Uses tqdm if available, otherwise simple counters
3. **State management**: Saves state every 10 issues for resumability
4. **Error handling**: 3-attempt retry with exponential backoff
5. **Logging**: Comprehensive logs to `logs/create_issues_TIMESTAMP.log`
6. **Colorful output**: ANSI colors (can be disabled with --no-color)
7. **File updates**: Updates github_issue.md files with created issue URLs

### Output Format

#### Dry-Run Summary
```
============================================================
DRY RUN MODE - No issues will be created
============================================================

Found 331 github_issue.md files
Total issues to create: 1,655

Breakdown by section:
  - 01-foundation: 210 issues (42 components)
  - 02-shared-library: 285 issues (57 components)
  - 03-tooling: 265 issues (53 components)
  - 04-first-paper: 415 issues (83 components)
  - 05-ci-cd: 220 issues (44 components)
  - 06-agentic-workflows: 260 issues (52 components)

Breakdown by type:
  - Plan: 331 issues
  - Test: 331 issues
  - Implementation: 331 issues
  - Packaging: 331 issues
  - Cleanup: 331 issues

Example issues (first 5):
[Shows detailed examples...]

Dry run complete. Run without --dry-run to create issues.
```

#### During Execution
```
Creating GitHub issues...
[1/1655] ✓ Created #123: [Plan] Create Base Directory - Design and Documentation
[2/1655] ✓ Created #124: [Test] Create Base Directory - Write Tests
[3/1655] ✓ Created #125: [Impl] Create Base Directory - Implementation
...
Progress: 10/1655 (0.6%) | Saved state to .issue_creation_state.json
...
[1655/1655] ✓ Created #1777: [Cleanup] Tutorial Clarity - Refactor and Finalize

============================================================
Summary
============================================================
Total issues created: 1,655
Total issues failed: 0
Files updated: 331

Time elapsed: XX minutes
```

## State Management

The script saves progress to `.issue_creation_state.json` every 10 issues:

```json
{
  "last_processed_index": 100,
  "created_issues": [
    {
      "title": "[Plan] ...",
      "issue_url": "https://github.com/mvillmow/ml-odyssey/issues/123",
      "file_path": "notes/plan/...",
      "created": true
    },
    ...
  ],
  "timestamp": "2025-11-06T21:59:18"
}
```

To resume after interruption:
```bash
python3 scripts/create_issues.py --resume
```

## Logging

Comprehensive logs are saved to `logs/create_issues_TIMESTAMP.log`:

```
2025-11-06 21:59:18,888 - INFO - Logging to /home/mvillmow/ml-odyssey/logs/create_issues_20251106_215918.log
2025-11-06 21:59:18,890 - INFO - Using repository: mvillmow/ml-odyssey
2025-11-06 21:59:19,100 - INFO - Found 331 github_issue.md files
2025-11-06 21:59:20,500 - INFO - Parsed 1655 issues
2025-11-06 21:59:20,501 - INFO - Creating issue 1/1655: [Plan] Create Base Directory
2025-11-06 21:59:21,234 - INFO - Created issue #123
2025-11-06 21:59:21,235 - INFO - Updated file: notes/plan/.../github_issue.md
...
```

## Recommended Execution Plan

### Step 1: Test with Dry-Run (Completed ✅)
```bash
python3 scripts/create_issues.py --dry-run
```
- Validates parsing of all 331 files
- Shows statistics and examples
- No actual GitHub API calls

### Step 2: Test with One Section
```bash
python3 scripts/create_issues.py --section 01-foundation
```
- Creates 210 issues for foundation section
- Validates actual GitHub issue creation
- Updates 42 github_issue.md files
- Takes ~5-10 minutes

### Step 3: Review and Validate
- Check created issues in GitHub UI
- Verify github_issue.md files were updated correctly
- Review logs for any errors

### Step 4: Create All Issues
```bash
python3 scripts/create_issues.py
```
- Creates all 1,655 issues
- Estimated time: ~30-45 minutes
- Monitor progress in real-time

### Step 5: Verify Completion
- Check that all github_issue.md files have issue URLs
- Verify all 1,655 issues exist in GitHub
- Review logs for any failures

## Error Handling

The script includes robust error handling:

1. **Retry logic**: Each failed issue creation is retried up to 3 times
2. **Exponential backoff**: Waits 1s, 2s, 4s between retries
3. **State saving**: Progress saved every 10 issues
4. **Resume capability**: Can resume from last saved state
5. **Error logging**: All errors logged with full details

## What Gets Updated

After each issue is created, the corresponding github_issue.md file is updated:

**Before**:
```markdown
**GitHub Issue URL**: [To be created]
```

**After**:
```markdown
**GitHub Issue URL**: https://github.com/mvillmow/ml-odyssey/issues/123
```

## Next Steps After Issue Creation

1. **Create PRs**: Create ~331 PRs (one per component) to commit the plans
2. **Organize Issues**: Use GitHub Projects to organize the 1,655 issues
3. **Create Milestones**: Group issues by top-level section
4. **Begin Implementation**: Start working through issues in order

## Notes

- The script auto-detects the repository from `git remote get-url origin`
- Labels are created automatically if they don't exist
- Issue bodies are written to temporary files and cleaned up after creation
- All operations are logged for auditing and debugging
