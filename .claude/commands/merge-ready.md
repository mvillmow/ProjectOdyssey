Check if PRs are ready to merge: $ARGUMENTS

For each PR number (or all open PRs if none specified):

1. Check CI status: `gh pr checks <pr>`
2. Check for merge conflicts: `gh pr view <pr> --json mergeable`
3. Check approval status: `gh pr view <pr> --json reviews`

Output a summary table:

| PR | Title | CI | Conflicts | Approvals | Ready? |

For PRs that are ready, suggest: `gh pr merge <pr> --rebase`
For PRs that are blocked, explain what's missing.
