---
name: mojo-memory-check
description: Verify memory safety in Mojo code including ownership, borrowing, and lifetime management. Use when reviewing code or debugging memory issues.
---

# Memory Safety Check Skill

Verify Mojo code follows memory safety rules.

## When to Use

- Reviewing code for memory safety
- Debugging memory issues
- Before merging code
- Performance testing reveals issues

## Memory Ownership

### Owned Parameters

```mojo
fn consume(owned data: Tensor):
    # Takes ownership, original is invalidated
    process(data)
    # data is destroyed here
```text

### Borrowed Parameters

```mojo
fn read_only(borrowed data: Tensor):
    # Read-only access, no ownership transfer
    let value = data[0]
    # data still valid in caller
```text

### Inout Parameters

```mojo
fn modify(inout data: Tensor):
    # Mutable reference
    data[0] = 42
    # Changes visible in caller
```text

## Common Issues

### 1. Use After Move

```mojo
# ❌ Wrong
fn bad():
    let data = Tensor()
    consume(data^)  # Move ownership
    print(data[0])  # Error: data moved!

# ✅ Correct
fn good():
    let data = Tensor()
    let copy = data  # Copy if needed
    consume(data^)
    print(copy[0])  # OK
```text

### 2. Lifetime Issues

```mojo
# ❌ Wrong
fn return_dangling() -> borrowed Tensor:
    let local = Tensor()
    return local  # Returns reference to local!

# ✅ Correct
fn return_owned() -> Tensor:
    let local = Tensor()
    return local^  # Move ownership
```text

### 3. Mutable Aliasing

```mojo
# ❌ Wrong
fn alias_problem(inout a: Tensor, borrowed b: Tensor):
    a[0] = b[0]  # If a and b alias, undefined!

# ✅ Correct
fn no_alias(inout a: Tensor, borrowed b: Tensor):
    # Ensure a and b don't alias
    if a.ptr() != b.ptr():
        a[0] = b[0]
```text

## Checklist

- [ ] No use-after-move
- [ ] No dangling references
- [ ] No mutable aliasing
- [ ] Proper lifetime management
- [ ] Ownership clear and documented

## Prod Fix Learnings (ML Odyssey)

From memory leaks in ExTensor views:

- **Refcounting**: Add `var _refcount: UnsafePointer[Int]`; `__copyinit__` increments, `__del__` decrements/free if 0.
- **Dummy Alloc Leaks**: In reshape/slice: `result._data.free()` before `result._data = self._data`; set `_is_view = True`.
- **Views**: Copy-based for safe shape/strides update, no orphan allocs.

See [docs/learnings.md](../docs/learnings.md#memory-leaks-reshapeslice) for details.

See Mojo ownership documentation for details.
