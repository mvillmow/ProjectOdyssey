# GitHub Issue Update Instructions

## Summary

This document contains instructions for updating ALL 331 `github_issue.md` files across the ML Odyssey project to include detailed, specific bodies for each of the 5 issue types (Plan, Test, Implementation, Packaging, Cleanup).

## What Has Been Done

1. **Created Update Script**: `/home/mvillmow/ml-odyssey/simple_update.py`
   - This Python script will automatically update all `github_issue.md` files
   - It extracts information from corresponding `plan.md` files
   - It generates detailed bodies for each issue type

2. **Manually Updated 2 Example Files** (to demonstrate the pattern):
   - `/home/mvillmow/ml-odyssey/notes/plan/01-foundation/01-directory-structure/01-create-papers-dir/github_issue.md`
   - `/home/mvillmow/ml-odyssey/notes/plan/02-shared-library/github_issue.md`

## How to Complete the Update

### Option 1: Run the Automated Script (Recommended)

Simply run the following command from the repository root:

```bash
cd /home/mvillmow/ml-odyssey
python3 simple_update.py
```

This will:
- Find all 331 `github_issue.md` files
- Extract component names from directory paths
- Read overview sections from corresponding `plan.md` files
- Generate detailed bodies for all 5 issue types
- Update each file in place
- Print progress as it goes
- Show a summary at the end

Expected output:
```
Found 331 github_issue.md files
[1/331] Updated: /home/mvillmow/ml-odyssey/notes/plan/...
[2/331] Updated: /home/mvillmow/ml-odyssey/notes/plan/...
...
================================================================================
Update Summary:
  Total files found: 331
  Successfully updated: 331
  Failed: 0
```

### Option 2: Verify the Updates

After running the script, you can verify a few random files to ensure they were updated correctly:

```bash
# Check a random file
cat notes/plan/03-tooling/github_issue.md

# Check another one
cat notes/plan/04-first-paper/02-model-implementation/github_issue.md

# Verify the count of properly formatted files
grep -l "## Planning Tasks" notes/plan/**/github_issue.md | wc -l
# Should output: 331
```

### Option 3: Manual Verification

Check that the new format includes:
- Detailed task lists for each issue type
- Component-specific overview from plan.md
- Proper links to plan.md files
- All 5 issue types (Plan, Test, Implementation, Packaging, Cleanup)

## New Format Structure

Each `github_issue.md` file now contains:

### 1. Plan Issue
- Objective and overview
- Planning task checklist
- Deliverables list
- Reference to plan.md

### 2. Test Issue
- Testing objectives (TDD focus)
- Testing task checklist
- Test type requirements
- Acceptance criteria

### 3. Implementation Issue
- Implementation objectives
- Implementation task checklist
- Implementation guidelines
- Success criteria

### 4. Packaging Issue
- Integration objectives
- Packaging task checklist
- Integration checks
- Validation requirements

### 5. Cleanup Issue
- Cleanup objectives
- Code quality checklist
- Documentation tasks
- Final validation

## Component Name Examples

The script generates human-readable component names from directory paths:

- `01-foundation/02-configuration-files/01-magic-toml` → "Foundation - Configuration Files - Magic Toml"
- `02-shared-library/01-core-operations` → "Shared Library - Core Operations"
- `04-first-paper/02-model-implementation/01-core-layers` → "First Paper - Model Implementation - Core Layers"

## Files

- **Update Script**: `simple_update.py` (ready to run)
- **Example Updated Files**:
  - `notes/plan/01-foundation/01-directory-structure/01-create-papers-dir/github_issue.md`
  - `notes/plan/02-shared-library/github_issue.md`

## Troubleshooting

If the script fails on any files, it will:
- Print the error message
- Continue processing other files
- Show failed files in the summary

Common issues:
- Missing `plan.md` file: Script will use generic overview
- Permissions: Ensure you have write access to notes/plan directory

## Next Steps

After running the update:

1. Review a few files to ensure quality
2. Commit the changes with a descriptive message:
   ```bash
   git add notes/plan/**/github_issue.md
   git commit -m "Update all github_issue.md files with detailed issue bodies"
   ```
3. The issues are now ready to be created in GitHub!

## Notes

- Total files to update: 331
- Files already updated: 2 (as examples)
- Files remaining: 329
- Estimated time: ~10-30 seconds (depending on system)
