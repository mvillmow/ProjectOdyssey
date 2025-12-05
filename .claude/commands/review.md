Review PR #$ARGUMENTS comprehensively.

1. Get PR details: `gh pr view $ARGUMENTS`
2. Get the diff: `gh pr diff $ARGUMENTS`
3. Check CI status: `gh pr checks $ARGUMENTS`
4. Review the changes for:
   - Code correctness and logic
   - Mojo syntax standards (out self in __init__, mut for mutating, List not DynamicVector)
   - Test coverage for new functionality
   - Documentation updates if needed
   - Memory safety (ownership, transfer operators)
5. Provide structured feedback with specific file:line references
6. Summarize: approve, request changes, or comment
