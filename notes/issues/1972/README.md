# Issue #1972: Fix Metrics Ownership from read to var Parameters

## Objective

Change function signatures in accuracy.mojo and confusion_matrix.mojo to accept `var` instead of `read`
for the `predictions` parameter. This allows ownership transfer with the `^` operator, which is required
for proper memory management in Mojo.

## Deliverables

- `/home/mvillmow/worktrees/1972-fix-metrics-ownership/shared/training/metrics/accuracy.mojo` - Updated argmax helper function signature
- `/home/mvillmow/worktrees/1972-fix-metrics-ownership/shared/training/metrics/confusion_matrix.mojo` - Updated argmax helper function signature

## Success Criteria

- [x] accuracy.mojo argmax function accepts `var tensor` parameter instead of `read`
- [x] confusion_matrix.mojo argmax function accepts `var tensor` parameter instead of `read`
- [x] Changes allow ownership transfer using `^` operator
- [x] Fixes "Cannot transfer out of an immutable reference" errors

## Implementation Notes

### Problem Analysis

The issue was that both files attempted to transfer ownership of the `predictions` parameter using the
`^` operator:

- **accuracy.mojo**: Lines 63, 303, 399 - Attempted `predictions^` but parameter was immutable
- **confusion_matrix.mojo**: Line 96 - Attempted `predictions^` but parameter was immutable

In Mojo, the `^` operator transfers ownership and can only be used on owned values (parameters declared
with `var`). Using it on immutable references (implicit `read` parameter convention) causes:
```
Error: Cannot transfer out of an immutable reference
```

### Solution Applied

Changed the argmax helper function signatures in both files:

**accuracy.mojo (line 93)**:
```mojo
# Before
fn argmax(tensor: ExTensor, axis: Int) raises -> ExTensor:

# After
fn argmax(var tensor: ExTensor, axis: Int) raises -> ExTensor:
```

**confusion_matrix.mojo (line 309)**:
```mojo
# Before
fn argmax(tensor: ExTensor) raises -> ExTensor:

# After
fn argmax(var tensor: ExTensor) raises -> ExTensor:
```

By declaring `tensor` as `var`, the parameter becomes an owned value. When called with `predictions^`,
the ownership of the tensor is transferred to the function, and the function can use it without
ownership restrictions.

### How It Fixes the Calls

The callers can now successfully use the `^` operator:

- **accuracy.mojo (line 63)**: `pred_classes = predictions^` - Transfers ownership to local variable
- **accuracy.mojo (line 303)**: `pred_classes = predictions^` - in per_class_accuracy
- **accuracy.mojo (line 399)**: `pred_classes = predictions^` - in AccuracyMetric.update
- **confusion_matrix.mojo (line 96)**: `pred_classes = predictions^` - Transfers ownership to local variable

This fixes the Top-Level Tests group failures related to metrics.

## References

- Related: [Mojo Language Review Specialist](../../../../.claude/agents/mojo-language-review-specialist.md)
- See: [CLAUDE.md - Mojo Syntax Standards](../../../../CLAUDE.md#mojo-syntax-standards-v0257)
- Memory Management: [Mojo Manual - Value Ownership](https://docs.modular.com/mojo/manual/values/ownership/)
