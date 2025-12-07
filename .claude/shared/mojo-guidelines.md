# Mojo Guidelines

Shared Mojo language guidelines for all agents. Reference this file instead of duplicating.

## When to Use Mojo vs Python

| Use Case | Language | Reason |
|----------|----------|--------|
| ML/AI implementations | Mojo (required) | Performance, type safety |
| Performance-critical code | Mojo (required) | SIMD, optimization |
| Subprocess output capture | Python (allowed) | Mojo v0.25.7 limitation |
| Regex processing | Python (allowed) | No Mojo stdlib support |
| GitHub API interaction | Python (allowed) | Library availability |

**Default**: Mojo unless technical limitation documented.

## Current Syntax (v0.25.7+)

### Parameter Conventions

| Convention | Use For | Example |
|------------|---------|---------|
| `out self` | Constructors | `fn __init__(out self, value: Int)` |
| `mut self` | Mutating methods | `fn modify(mut self)` |
| `read` (default) | Read-only access | `fn get(self) -> Int` |
| `var` + `^` | Ownership transfer | `fn consume(var data: List[T])` |

### Deprecated Patterns

| Wrong | Correct | Notes |
|-------|---------|-------|
| `borrowed self` | `self` | Deprecated keyword |
| `inout self` | `mut self` | Deprecated keyword |
| `@value` | `@fieldwise_init` + traits | Add `(Copyable, Movable)` |
| `DynamicVector[T]` | `List[T]` | Use `.append()` not `.push_back()` |
| `-> (T1, T2)` | `-> Tuple[T1, T2]` | Explicit tuple type |

### Function Definitions

**Use `fn`** for:

- Performance-critical code (compile-time optimization)
- Functions with explicit type annotations
- SIMD/vectorized operations
- Production APIs

**Use `def`** for:

- Python-compatible functions
- Quick prototypes
- Dynamic typing needed

### Struct Patterns

```mojo
# With @fieldwise_init (recommended for simple structs)
@fieldwise_init
struct Point(Copyable, Movable):
    var x: Float32
    var y: Float32

# Manual constructor (for complex initialization)
struct Tensor(Copyable, Movable):
    var data: DTypePointer[DType.float32]
    var shape: List[Int]

    fn __init__(out self, shape: List[Int]):
        self.shape = shape
        # ... initialization logic
```

## Pre-Commit Checklist

- [ ] All `__init__` use `out self` (not `mut self`)
- [ ] No `inout` keyword (use `mut`)
- [ ] No `@value` decorator (use `@fieldwise_init` + traits)
- [ ] All List/Dict returns use `^` transfer operator
- [ ] Space after `var` keyword: `var a` not `vara`

See [mojo-anti-patterns.md](mojo-anti-patterns.md) for common mistakes.
See [CLAUDE.md](../../CLAUDE.md#mojo-syntax-standards-v0257) for complete reference.
