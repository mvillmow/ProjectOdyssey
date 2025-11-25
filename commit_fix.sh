#!/bin/bash
cd /home/mvillmow/worktrees/1978-fix-early-stopping

# Add the modified file
git add shared/training/callbacks.mojo

# Create commit with exact message format
git commit -m "fix(training): correct min_delta logic in EarlyStopping callback

EarlyStopping callback had inverted logic for min_delta threshold.
Small improvements below min_delta were incorrectly resetting the
patience counter instead of counting toward early stopping.

Fixed comparison logic to correctly identify improvements that meet
or exceed the threshold. Improvements below min_delta now properly
count toward early stopping trigger.

This fixes Training: Callbacks test group failures.

Closes #1978

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Show the commit
git log -1 --stat
