Fix CI failures for PR $ARGUMENTS.

1. Get the PR number (use $ARGUMENTS or detect from current branch if empty)
2. Check CI status: `gh pr checks <pr>`
3. If failing, get failed run logs: `gh run view <run-id> --log-failed`
4. Identify the failing test/build step
5. Read the relevant source files
6. Fix the issue
7. Commit and push the fix
8. Wait for CI and verify it passes

If multiple PRs provided, fix each one sequentially.
