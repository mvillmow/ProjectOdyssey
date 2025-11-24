---
name: quality-fix-formatting
description: Automatically fix code formatting issues using mojo format, markdownlint --fix, and pre-commit hooks. Use when formatting checks fail or before committing code.
---

# Fix Formatting Skill

This skill automatically fixes code formatting issues across all file types.

## When to Use

- User asks to fix formatting (e.g., "fix all formatting issues")
- Pre-commit checks fail due to formatting
- Before committing code
- After writing new code
- CI formatting checks fail

## Auto-Fix Capabilities

### Mojo Code (100% auto-fix)

```bash
# Format all Mojo files
mojo format src/**/*.mojo

# Automatically fixes
# - Indentation
# - Spacing around operators
# - Line wrapping
# - Blank lines
```text

### Markdown (Partial auto-fix)

```bash
# Fix markdown issues
npx markdownlint-cli2 --fix "**/*.md"

# Auto-fixes
# - Some blank line issues
# - Some spacing issues
#
# Manual fixes needed for
# - Missing language in code blocks
# - Line length (need to reflow text)
```text

### Pre-commit Hooks (100% auto-fix)

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Auto-fixes
# - Trailing whitespace
# - Missing final newline
# - Mixed line endings
# - YAML formatting
```text

## Usage

### Fix Everything

```bash
# Run all formatters
./scripts/fix_all_formatting.sh

# This
# 1. Formats all Mojo files
# 2. Fixes markdown issues
# 3. Runs pre-commit hooks
# 4. Reports what was fixed
```text

### Fix Specific File Type

```bash
# Mojo only
./scripts/fix_formatting.sh --mojo

# Markdown only
./scripts/fix_formatting.sh --markdown

# Pre-commit only
./scripts/fix_formatting.sh --precommit
```text

### Fix Specific File

```bash
# Fix single Mojo file
mojo format src/tensor.mojo

# Fix single markdown file
npx markdownlint-cli2 --fix README.md

# Fix by pre-commit
pre-commit run --files src/tensor.mojo
```text

## Workflow

### When Formatting Fails

```bash
# 1. Run fix-all
./scripts/fix_all_formatting.sh

# 2. Review changes
git diff

# 3. Stage changes
git add .

# 4. Commit
git commit -m "fix: apply code formatting"
```text

### Before Committing

```bash
# 1. Fix formatting proactively
./scripts/fix_all_formatting.sh

# 2. Add files
git add .

# 3. Commit (pre-commit will run)
git commit -m "feat: new feature"
```text

### After CI Failure

```bash
# 1. Pull latest changes
git pull

# 2. Fix formatting
./scripts/fix_all_formatting.sh

# 3. Commit and push
git add .
git commit --amend --no-edit
git push --force-with-lease
```text

## Common Fixes

### Mojo Formatting

### Before

```mojo
fn add(x:Int,y:Int)->Int:
    return x+y
```text

### After

```mojo
fn add(x: Int, y: Int) -> Int:
    return x + y
```text

### Markdown Formatting

### Before

````markdown
Some text before.
```text

code block

```text
Some text after.
````

### After

````markdown
Some text before.

```text

code block

```text
Some text after.
````

### Trailing Whitespace

### Before

```text
line with trailing spaces
another line
```text

### After

```text
line with trailing spaces
another line
```text

## Manual Fixes Required

Some issues need manual intervention:

### Markdown Language Tags

````markdown
# Before (need to add manually)
```text

code here

```text
# After
```python

code here

```text
````

### Line Length

```markdown
# Before (too long)
This is a very long line that exceeds the 120 character limit and should be broken into multiple lines.

# After (manually break)
This is a very long line that exceeds the 120 character limit and should be
broken into multiple lines.
```text

## Error Handling

### Syntax Errors

```text
Error: Cannot format file.mojo due to syntax error
```text

**Fix**: Correct syntax errors before formatting

### Permission Denied

```text
Error: Permission denied: file.mojo
```text

**Fix**: Check file permissions

### Merge Conflicts

```text
Error: File contains merge conflict markers
```text

**Fix**: Resolve merge conflicts first

## Examples

### Fix all formatting:

```bash
./scripts/fix_all_formatting.sh
```text

### Fix only Mojo:

```bash
mojo format src/**/*.mojo
```text

### Fix only markdown:

```bash
npx markdownlint-cli2 --fix "**/*.md"
```text

### Fix and commit:

```bash
./scripts/fix_all_formatting.sh
git add .
git commit -m "fix: apply code formatting"
```text

## Scripts Available

- `scripts/fix_all_formatting.sh` - Fix all formatting
- `scripts/fix_formatting.sh` - Fix specific types
- `scripts/check_formatting.sh` - Check without fixing

## Integration with CI

CI checks formatting but doesn't fix:

```yaml
- name: Check Formatting
  run: |
    mojo format --check src/**/*.mojo
    npx markdownlint-cli2 "**/*.md"
```text

If CI fails, fix locally and push.

## Best Practices

1. **Fix before commit** - Always format before committing
1. **Use pre-commit** - Let hooks auto-fix on commit
1. **Review changes** - Check what was formatted
1. **Commit separately** - Formatting changes in separate commit
1. **Don't bypass** - Don't use `--no-verify` to skip formatting

## Prod Fix Learnings (ML Odyssey)

**List Constructor Anti-Pattern** (8 bugs fixed):

- **Never**: `List[Int](n)` then `list[i] = val` (undefined size → crash).
- **Always**: `List[Int]()` + `.append(val)`.

Affected: shape.mojo (reshape/squeeze/unsqueeze/concat 4x), accuracy/confusion/DataLoader.

Add lint rule in pre-commit.

See [docs/learnings.md](../docs/learnings.md#list-constructor-bugs-8-instances).

## Auto-Fix Summary

| Tool | Auto-Fix | Manual Needed |
|------|----------|---------------|
| mojo format | 100% | None |
| markdownlint | 70% | Language tags, line length |
| pre-commit | 100% | None |

See `.pre-commit-config.yaml` for formatting configuration.
