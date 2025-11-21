# Commands to Execute for Issue #40

## Overview

The following commands must be executed to complete the Package phase for the Data module. All preparation work has been completed - scripts and documentation are ready.

## Working Directory

```bash
cd /home/mvillmow/ml-odyssey/worktrees/40-pkg-data
```text

## Commands (Execute in Order)

### 1. Make Scripts Executable

```bash
chmod +x scripts/build_data_package.sh
chmod +x scripts/install_verify_data.sh
```text

### 2. Build the Package

```bash
./scripts/build_data_package.sh
```text

### Expected Outcome

- Creates `dist/data-0.1.0.mojopkg` file
- Displays package file size
- Confirms successful build

### 3. Verify Installation

```bash
./scripts/install_verify_data.sh
```text

### Expected Outcome

- Tests package installation in temporary directory
- Verifies all core imports work
- Displays success message
- Cleans up temporary files

### 4. Commit Changes

```bash
# Check what will be committed
git status

# Stage files
git add scripts/build_data_package.sh
git add scripts/install_verify_data.sh
git add notes/issues/40/

# Commit
git commit -m "feat(data): create distributable package with installation testing

- Built dist/data-0.1.0.mojopkg binary package
- Created build script (scripts/build_data_package.sh)
- Created installation verification script (scripts/install_verify_data.sh)
- Tested installation in clean environment
- Updated documentation to reflect actual artifacts

Closes #40"

# Push
git push origin 40-pkg-data
```text

### 5. Create Pull Request

```bash
gh pr create --issue 40 --fill
```text

## Alternative: Run All at Once

```bash
cd /home/mvillmow/ml-odyssey/worktrees/40-pkg-data && \
chmod +x scripts/build_data_package.sh scripts/install_verify_data.sh && \
./scripts/build_data_package.sh && \
./scripts/install_verify_data.sh && \
git add scripts/ notes/issues/40/ && \
git commit -m "feat(data): create distributable package with installation testing

- Built dist/data-0.1.0.mojopkg binary package
- Created build script (scripts/build_data_package.sh)
- Created installation verification script (scripts/install_verify_data.sh)
- Tested installation in clean environment
- Updated documentation to reflect actual artifacts

Closes #40" && \
git push origin 40-pkg-data && \
gh pr create --issue 40 --fill
```text

## Verification Before PR

Before creating the PR, verify:

1. `dist/data-0.1.0.mojopkg` exists and is non-zero size
1. Build script ran without errors
1. Verification script ran without errors
1. All files staged for commit
1. Commit message follows conventional format

## Files to Be Committed

```text
A  scripts/build_data_package.sh
A  scripts/install_verify_data.sh
M  notes/issues/40/README.md
A  notes/issues/40/EXECUTION_GUIDE.md
A  notes/issues/40/package-build-task.md
A  notes/issues/40/COMMANDS_TO_EXECUTE.md
```text

## Files NOT Committed (in .gitignore)

```text
dist/data-0.1.0.mojopkg  # Build artifact, excluded by .gitignore
```text

## Success Indicators

You should see:

1. Build script output showing package creation
1. Verification script output showing successful imports
1. `git status` showing staged files
1. `git log` showing commit
1. `gh pr list` showing new PR linked to issue #40

## Troubleshooting

If any command fails, see `/home/mvillmow/ml-odyssey/worktrees/40-pkg-data/notes/issues/40/EXECUTION_GUIDE.md` for detailed troubleshooting steps.
