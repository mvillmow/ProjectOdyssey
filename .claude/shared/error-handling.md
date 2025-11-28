# Error Handling

Shared error handling protocols for all agents. Reference this file instead of duplicating.

## Retry Strategy

- **Max Attempts**: 3 retries for failed operations
- **Backoff**: Exponential backoff (1s, 2s, 4s between attempts)
- **Scope**: Apply to delegation failures, not system errors

## Timeout Handling

- **Max Wait**: 5 minutes for delegated work to complete
- **On Timeout**: Escalate to parent with context about what timed out
- **Check Interval**: Poll for completion every 30 seconds

## Conflict Resolution

When receiving conflicting guidance:
1. Attempt to resolve conflicts based on specifications and priorities
2. If unable to resolve: escalate to parent level with full context
3. Document the conflict and resolution in status updates

## Failure Modes

### Partial Failure
Some delegated work succeeds, some fails.
- **Action**: Complete successful parts, escalate failed parts

### Complete Failure
All attempts at delegation fail.
- **Action**: Escalate immediately to parent with failure details

### Blocking Failure
Cannot proceed without resolution.
- **Action**: Escalate immediately, do not retry

## Loop Detection

- **Pattern**: Same operation attempted 3+ times with same result
- **Action**: Break the loop, escalate with loop context
- **Prevention**: Track attempts per unique task

## Escalation Triggers

Escalate errors when:
- All retry attempts exhausted
- Timeout exceeded
- Unresolvable conflicts detected
- Critical blocking issues found
- Loop detected in delegation chain

## Error Reporting Format

```markdown
## Error Report

**Agent**: [agent-name]
**Issue**: #[issue-number]
**Error Type**: [timeout|conflict|failure|loop]

### What Happened
[Brief description]

### What Was Tried
1. [Attempt 1]
2. [Attempt 2]
3. [Attempt 3]

### What's Needed
[Specific help required]
```
