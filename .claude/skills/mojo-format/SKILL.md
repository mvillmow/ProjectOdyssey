---
name: mojo-format
description: Format Mojo code files using the mojo format command to ensure consistent code style. Use when preparing code for commit or when code formatting checks fail.
---

# Mojo Format Skill

This skill formats Mojo code files using the official `mojo format` command.

## When to Use

- User asks to format Mojo code (e.g., "format the Mojo files")
- Pre-commit hook reports formatting issues
- Before committing Mojo code
- Code review requests formatting fixes

## Usage

### Format Single File

```bash
# Format a specific file
mojo format path/to/file.mojo

# Format with check-only (no changes)
mojo format --check path/to/file.mojo
```

### Format Multiple Files

```bash
# Format all Mojo files in directory
./scripts/format_mojo.sh src/

# Format all Mojo files in project
./scripts/format_mojo.sh .

# Check formatting without changes
./scripts/format_mojo.sh --check .
```

### Pre-commit Integration

The project uses pre-commit hooks to automatically format Mojo files:

```bash
# Pre-commit will auto-format on commit
git commit -m "message"

# Run pre-commit manually
pre-commit run mojo-format --all-files
```

## Mojo Format Behavior

### What It Formats

- **Indentation** - Consistent 4-space indentation
- **Line length** - Wraps long lines
- **Spacing** - Consistent spacing around operators
- **Blank lines** - Standardizes blank line usage
- **Comments** - Preserves comments, formats spacing

### What It Doesn't Change

- **Logic** - No semantic changes
- **Names** - Variable/function names unchanged
- **Comments content** - Only spacing adjusted

## Format Standards

### Function Definitions

```mojo
# Before formatting
fn add(x:Int,y:Int)->Int:
    return x+y

# After formatting
fn add(x: Int, y: Int) -> Int:
    return x + y
```

### Struct Definitions

```mojo
# Before formatting
struct Tensor[dtype:DType,rank:Int]:
    var data:DTypePointer[dtype]

# After formatting
struct Tensor[dtype: DType, rank: Int]:
    var data: DTypePointer[dtype]
```

## Scripted Formatting

### Format All Mojo Files

```bash
./scripts/format_mojo.sh

# This finds and formats all .mojo and .🔥 files
```

### Check Formatting (CI)

```bash
# Check if files need formatting (exit code 1 if changes needed)
./scripts/format_mojo.sh --check

# Used in CI to verify formatting
```

## Error Handling

- **Syntax errors**: `mojo format` will report but not fix syntax errors
- **File not found**: Verify file path is correct
- **Permission denied**: Check file permissions
- **Mojo not installed**: Install Mojo via `pixi` or Magic

## Integration with Workflow

### Before Commit

```bash
# Format all changed Mojo files
git diff --name-only --cached | grep '\.mojo$' | xargs -r mojo format

# Or let pre-commit handle it
git commit -m "message"  # pre-commit auto-formats
```

### CI Validation

Pre-commit CI workflow checks formatting:

```yaml
- repo: local
  hooks:
    - id: mojo-format
      name: Mojo Format
      entry: mojo format
      language: system
      files: \.(mojo|🔥)$
```

## Examples

**Format single file:**

```bash
mojo format src/tensor.mojo
```

**Format directory:**

```bash
./scripts/format_mojo.sh src/
```

**Check formatting:**

```bash
./scripts/format_mojo.sh --check .
```

**Format changed files:**

```bash
git diff --name-only | grep '\.mojo$' | xargs -r mojo format
```

## Scripts Available

- `scripts/format_mojo.sh` - Format all Mojo files
- `scripts/check_mojo_format.sh` - Check formatting without changes

## Best Practices

1. **Format before commit** - Run formatting before every commit
2. **Use pre-commit** - Let hooks handle formatting automatically
3. **Check in CI** - Verify formatting in CI pipeline
4. **Consistent style** - Don't manually reformat after `mojo format`
5. **Trust the formatter** - Accept formatter's decisions

See `.pre-commit-config.yaml` for pre-commit hook configuration.
