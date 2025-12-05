Run tests: $ARGUMENTS

If no arguments provided:

- Detect changed files from `git status` and `git diff --name-only`
- Identify which test files correspond to changed modules
- Run tests for those modules

If path provided:

- If it's a directory: `mojo test -I . $ARGUMENTS`
- If it's a file: `mojo test -I . $ARGUMENTS`

Show test results summary with pass/fail counts.
After tests complete, report any failures with file:line references.
