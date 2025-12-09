# Safety Hooks for Claude Code

This directory contains safety hooks that validate commands before execution to prevent dangerous operations.

## Purpose

Prevent accidental or malicious deletion of critical files and directories, including:

- .git directory and files
- Files outside the project directory
- Dangerous patterns like `rm -rf /`

## Files

- **pre-bash-exec.sh**: Pre-execution hook that validates `rm` commands
- **test-safety-hooks.sh**: Comprehensive test suite for the safety hooks
- **README.md**: This file

## How It Works

The `pre-bash-exec.sh` hook is configured in `.claude/settings.local.json` to run before every Bash command execution. It validates `rm` commands against a set of dangerous patterns and blocks them if they match.

### Blocked Patterns

1. **Root directory deletion**: `rm -rf /` or variations
2. **.git deletion**: Any `rm` command targeting `.git` directory or files
3. **Outside project deletion**: `rm` with absolute paths outside the project root
4. **Parent directory escaping**: Paths using `../` that escape the project directory

### Allowed Operations

- `rm` commands within the project directory (relative or absolute paths)
- `rm` commands with relative paths that stay within project bounds
- All non-`rm` commands pass through unchanged

### Warnings

- **sudo rm**: Commands using `sudo` with `rm` generate a warning but are not blocked (with current implementation)

## Configuration

The hook is configured in `.claude/settings.local.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/pre-bash-exec.sh \"$ARGUMENTS\"",
            "timeout": 5,
            "statusMessage": "Validating command safety..."
          }
        ]
      }
    ]
  }
}
```

## Testing

Run the test suite to validate the hooks:

```bash
bash .claude/hooks/test-safety-hooks.sh
```

The test suite includes:

- 7 dangerous commands (should be blocked)
- 6 safe commands (should pass)
- 3 edge cases

## Error Messages

When a dangerous command is blocked, you'll see clear error messages:

```text
ERROR: Blocked dangerous command - attempting to delete root directory
Command: rm -rf /
```

```text
ERROR: Blocked dangerous command - attempting to delete .git directory or files
Command: rm -rf .git
```

```text
ERROR: Blocked dangerous command - attempting to delete files outside project directory
Path: /etc/passwd
Project root: /home/user/project
Command: rm /etc/passwd
```

## Implementation Details

### Validation Flow

1. Hook receives the command to be executed
2. Check if command contains `rm`
3. If yes, validate against dangerous patterns:
   - Extract paths from rm command
   - Check for root directory deletion
   - Check for .git targeting
   - Verify paths are within project
4. Return exit code:
   - 0 = safe, proceed with execution
   - 1 = dangerous, block execution

### Project Root Detection

The hook determines the project root using:

```bash
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
```

This ensures paths are validated against the actual project boundary.

### Path Resolution

Paths are resolved to absolute paths before validation to handle:

- Relative paths (`./file.txt`, `../file.txt`)
- Absolute paths (`/home/user/project/file.txt`)
- Symlinks (resolved to actual target)

## Customization

To customize the blocked patterns, edit `.claude/hooks/pre-bash-exec.sh` and modify the `validate_rm_command()` function.

### Adding New Patterns

Add additional validation checks:

```bash
# Pattern 5: Block deletion of specific files
if echo "$cmd" | grep -qE '\brm\b.*important-file\.txt'; then
    echo "ERROR: Blocked - attempting to delete important file" >&2
    return 1
fi
```

### Adjusting Warnings

To block (instead of warn) on sudo:

```bash
if echo "$cmd" | grep -qE '\bsudo\s+rm\b'; then
    echo "ERROR: Blocked - sudo rm is not allowed" >&2
    return 1  # Changed from warning to error
fi
```

## Maintenance

- Test hooks after modifying validation logic
- Keep error messages clear and actionable
- Document any new patterns added
- Run test suite in CI to catch regressions

## Security Considerations

### Defense in Depth

This hook is ONE layer of security. It does NOT replace:

- File system permissions
- Backup procedures
- Code review
- User awareness

### Limitations

- Hooks can be disabled in settings
- Skilled users can bypass hooks
- Complex commands may evade pattern matching
- Not a substitute for proper access control

### Best Practices

1. **Keep hooks enabled**: Don't disable without good reason
2. **Review settings changes**: Audit `.claude/settings.local.json` modifications
3. **Test thoroughly**: Run test suite after changes
4. **Document exceptions**: If you need to bypass a hook, document why

## Troubleshooting

### Hook not running

Check configuration:

```bash
cat .claude/settings.local.json | grep -A 10 hooks
```

Verify hook script exists and is executable:

```bash
ls -la .claude/hooks/pre-bash-exec.sh
```

### False positives

If a safe command is blocked:

1. Review the error message
2. Check if the path is truly within project
3. Consider adjusting validation logic
4. Document the case and update patterns

### False negatives

If a dangerous command passes:

1. Add the pattern to the test suite
2. Update validation logic to catch it
3. Run tests to verify
4. Commit the fix

## Support

For issues or questions:

1. Check this README
2. Review test cases in `test-safety-hooks.sh`
3. Examine validation logic in `pre-bash-exec.sh`
4. Create an issue in the project repository

## License

Same as the main project (see root LICENSE file).
