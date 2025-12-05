Check CI status for PRs: $ARGUMENTS

For each PR number provided (or all open PRs if none specified):

1. Run `gh pr checks <pr>` to get current status
2. Summarize: PR number, title, status (pass/fail/pending)
3. If any failing, show which check failed

Output as a concise table:

| PR | Title | Status | Failed Check |
