# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML Odyssey is a Mojo-based AI research platform for reproducing classic research papers. The project uses a
comprehensive 4-level hierarchical planning structure with automated GitHub issue creation.

**Current Status**: Planning phase - repository structure and GitHub issues are being established before implementation begins.

## Working with Agents

This project uses a hierarchical agent system for all development work. **Always use agents** as the primary
method for completing tasks.

### Agent Hierarchy

See [agents/hierarchy.md](agents/hierarchy.md) for the complete agent hierarchy including:

- 6-level hierarchy (L0 Chief Architect → L5 Junior Engineers)
- Model assignments (Opus, Sonnet, Haiku)
- All 44 agents with roles and responsibilities

### Key Agent Principles

1. **Always start with orchestrators** for new section work
1. **All outputs** must be posted as comments on the GitHub issue
1. **Link all PRs** to issues using `gh pr create --issue <number>` or "Closes #123" in description
1. **Minimal changes only** - smallest change that solves the problem
1. **No scope creep** - focus only on issue requirements
1. **Reply to each review comment** with `✅ Fixed - [brief description]`
1. **Delegate to skills** - Use "Use the X skill to..." pattern for automation

### Skill Delegation Patterns

Agents delegate to skills for automation using five standard patterns:

**Pattern 1: Direct Delegation** - Agent needs specific automation

```markdown
Use the `skill-name` skill to [action]:
- **Invoke when**: [trigger condition]
- **The skill handles**: [specific automation]
```text

**Pattern 2: Conditional Delegation** - Agent decides based on conditions

```markdown
If [condition]:
  - Use the `skill-name` skill to [action]
Otherwise:
  - [alternative approach]
```text

**Pattern 3: Multi-Skill Workflow** - Agent orchestrates multiple skills

```markdown
To accomplish [goal]:
1. Use the `skill-1` skill to [step 1]
2. Use the `skill-2` skill to [step 2]
3. Review results and [decision]
```text

**Pattern 4: Skill Selection** - Orchestrator chooses skill based on analysis

```markdown
Analyze [context]:
- If [scenario A]: Use `skill-A`
- If [scenario B]: Use `skill-B`
```text

**Pattern 5: Background vs Foreground** - Distinguishing automatic vs explicit invocation

```markdown
Background automation: `ci-run-precommit` (runs automatically)
Foreground tasks: `gh-create-pr-linked` (invoke explicitly)
```text

**Available Skills** (82 total across 11 categories):

- **GitHub**: gh-review-pr, gh-fix-pr-feedback, gh-create-pr-linked, gh-check-ci-status, gh-implement-issue, gh-reply-review-comment, gh-get-review-comments, gh-batch-merge-by-labels, verify-pr-ready
- **Worktree**: worktree-create, worktree-cleanup, worktree-switch, worktree-sync
- **Phase Workflow**: phase-plan-generate, phase-test-tdd, phase-implement, phase-package, phase-cleanup
- **Mojo**: mojo-format, mojo-test-runner, mojo-build-package, mojo-simd-optimize, mojo-memory-check, mojo-type-safety, mojo-lint-syntax, validate-mojo-patterns, check-memory-safety, analyze-simd-usage
- **Agent System**: agent-validate-config, agent-test-delegation, agent-run-orchestrator, agent-coverage-check, agent-hierarchy-diagram
- **Documentation**: doc-generate-adr, doc-issue-readme, doc-validate-markdown, doc-update-blog
- **CI/CD**: ci-run-precommit, ci-validate-workflow, ci-fix-failures, ci-package-workflow, ci-analyze-failure-logs, build-run-local
- **Plan**: plan-regenerate-issues, plan-validate-structure, plan-create-component
- **Quality**: quality-run-linters, quality-fix-formatting, quality-security-scan, quality-coverage-report, quality-complexity-check
- **Testing & Analysis**: test-diff-analyzer, extract-test-failures, generate-fix-suggestions, track-implementation-progress
- **Review**: review-pr-changes, create-review-checklist

See `.claude/skills/` for complete implementations. Skills use YAML frontmatter with `mcp_fallback` for MCP integration.

### Key Development Principles

1. KISS - *K*eep *I*t *S*imple *S*tupid -> Don't add complexity when a simpler solution works
1. YAGNI - *Y*ou *A*in't *G*onna *N*eed *I*t -> Don't add things until they are required
1. TDD - *T*est *D*riven *D*evelopment -> Write tests to drive the implementation
1. DRY - *D*on't *R*epeat *Y*ourself -> Don't duplicate functionality, data structures, or algorithms
1. SOLID - *S**O**L**I**D* ->
  . Single Responsibility
  . Open-Closed
  . Liskov Substitution
  . Interface Segregation
  . Dependency Inversion
1. Modularity - Develop independent modules through well defined interfaces
1. POLA - *P*rinciple *O*f *L*east *A*stonishment - Create intuitive and predictable interfaces to not surprise users

Relevant links:

- [Core Principles of Software Development](https://softjourn.com/insights/core-principles-of-software-development)
- [7 Common Programming Principles](https://www.geeksforgeeks.org/blogs/7-common-programming-principles-that-every-developer-must-follow/)
- [Software Development Principles](https://coderower.com/blogs/software-development-principles-software-engineering)
- [Clean Coding Principles](https://www.pullchecklist.com/posts/clean-coding-principles)

### Documentation Rules

- **Issue-specific outputs**: Post as comments on the GitHub issue using `gh issue comment <number>`
- **Developer documentation**: `/docs/dev/` (architectural decisions, design docs)
- **Team guides**: `/agents/` (quick start, hierarchy, templates)
- **Never duplicate** documentation across locations - link instead
- See `.claude/shared/github-issue-workflow.md` for GitHub issue read/write patterns

### Language Preference

#### Mojo First - With Pragmatic Exceptions

**Default to Mojo** for ALL ML/AI implementations:

- ✅ Neural network implementations (forward/backward passes, layers)
- ✅ Training loops and optimization algorithms
- ✅ Tensor operations and SIMD kernels
- ✅ Performance-critical data processing
- ✅ Type-safe model components
- ✅ Gradient computation and backpropagation
- ✅ Model inference engines

**Use Python for Automation** when technical limitations require it:

- ✅ Subprocess output capture (Mojo v0.25.7 limitation - cannot capture stdout/stderr)
- ✅ Regex-heavy text processing (no Mojo regex support in stdlib)
- ✅ GitHub API interaction via Python libraries (`gh` CLI, REST API)
- ⚠️ **MUST document justification** (see ADR-001 for header template)

**Rule of Thumb** (Decision Tree):

1. **ML/AI implementation?** → Mojo (required)
1. **Automation needing subprocess output?** → Python (allowed, document why)
1. **Automation needing regex?** → Python (allowed, document why)
1. **Interface with Python-only libraries?** → Python (allowed, document why)
1. **Everything else?** → Mojo (default)

### Why Mojo for ML/AI

- Performance: Faster for ML workloads
- Type safety: Catch errors at compile time
- Memory safety: Built-in ownership and borrow checking
- SIMD optimization: Parallel tensor operations
- Future-proof: Designed for AI/ML from the ground up

### Why Python for Automation

- Mojo's subprocess API lacks exit code access (causes silent failures)
- Regex support not production-ready (mojo-regex is alpha stage)
- Python is the right tool for automation - not a temporary workaround

**See**: [ADR-001: Language Selection for Tooling](docs/adr/ADR-001-language-selection-tooling.md) for complete language selection strategy, technical evidence (test results), and justification requirements

See `/agents/README.md` for complete agent documentation and `/agents/hierarchy.md` for visual hierarchy.

## Delegation to Agent Hub

.claude/ is the centralized location for agentic descriptions and SKILLs. Sub-agents reference `.claude/agents/*.md` and `.claude/skills/*.md` for roles, capabilities, and prod fix learnings.

### Shared Reference Files

All agents and skills reference these shared files to avoid duplication:

| File | Purpose |
|------|---------|
| `.claude/shared/common-constraints.md` | Minimal changes principle, scope discipline |
| `.claude/shared/documentation-rules.md` | Output locations, before-starting checklist |
| `.claude/shared/pr-workflow.md` | PR creation, verification, review responses |
| `.claude/shared/mojo-guidelines.md` | Mojo v0.25.7+ syntax, parameter conventions |
| `.claude/shared/mojo-anti-patterns.md` | 64+ test failure patterns from PRs |
| `.claude/shared/error-handling.md` | Retry strategy, timeout handling, escalation |

### MCP Integration

Skills with `mcp_fallback: github` in their YAML frontmatter automatically fall back to the GitHub MCP server when available. MCP servers are configured in `.claude/settings.local.json`.

### Mojo Syntax Standards (v0.25.7+)

**CRITICAL**: Always use current Mojo syntax. The following patterns are DEPRECATED or INCORRECT:

#### ❌ DEPRECATED: `inout` keyword → ✅ USE: `mut`

**WRONG**:
```mojo
fn __init__(inout self, value: Int):
    self.value = value

fn modify(inout self):
    self.value += 1
```text

**CORRECT**:

```mojo
fn __init__(mut self, value: Int):
    self.value = value

fn modify(mut self):
    self.value += 1
```text

#### ❌ DEPRECATED: `@value` decorator → ✅ USE: `@fieldwise_init` + traits

**WRONG**:

```mojo
@value
struct Transform:
    var name: String
```text

**CORRECT**:

```mojo
@fieldwise_init
struct Transform(Copyable, Movable):
    var name: String
```text

#### ❌ NON-EXISTENT: `DynamicVector` → ✅ USE: `List`

**WRONG**:

```mojo
from collections.vector import DynamicVector

var values = DynamicVector[Int](10)
values.push_back(42)
```text

**CORRECT**:

```mojo
var values = List[Int](10)
values.append(42)
```text

#### ❌ INVALID: Tuple return syntax `-> (T1, T2)` → ✅ USE: `Tuple[T1, T2]`

**WRONG**:

```mojo
fn compute() -> (Float32, Float32):
    return (1.0, 2.0)
```text

**CORRECT**:

```mojo
fn compute() -> Tuple[Float32, Float32]:
    return Tuple[Float32, Float32](1.0, 2.0)
```text

#### ✅ CORRECT: Parameter Conventions

**Mojo v0.25.7+ parameter types**:

1. **`read`** (default) - Immutable reference:

```mojo
fn process(data: ExTensor):  # read is implicit
    print(data.shape)
```text

1. **`mut`** - Mutable reference (replaces `inout`):

```mojo
fn modify(mut data: ExTensor):
    data._fill_zero()
```text

1. **`var`** - Owned value (takes ownership):

```mojo
fn consume(var data: ExTensor):
    data += 1  # Owns the data, caller loses access
```text

1. **`ref`** - Parametric reference (advanced):

```mojo
fn generic_ref[mutability: Bool](ref [mutability] data: ExTensor):
    # Can be mutable or immutable based on parameter
```text

#### ✅ CORRECT: Struct Initialization Patterns

**With `@fieldwise_init` (recommended for simple structs)**:

```mojo
@fieldwise_init
struct Point(Copyable, Movable):
    var x: Float32
    var y: Float32

var p = Point(1.0, 2.0)  # Auto-generated constructor
```text

**Manual constructor (for complex initialization)**:

```mojo
struct Tensor(Copyable, Movable):
    var data: DTypePointer[DType.float32]
    var shape: List[Int]

    fn __init__(mut self, shape: List[Int]):
        self.shape = shape
        var size = 1
        for dim in shape:
            size *= dim
        self.data = DTypePointer[DType.float32].alloc(size)
```text

#### ✅ CORRECT: Common Mojo Patterns

**Loop with mutable references**:

```mojo
var list = List[Int](1, 2, 3)
for ref item in list:  # Use 'ref' to mutate
    item = item * 2
```text

**Ownership transfer with `^`**:

```mojo
fn take_ownership(var data: String):
    print(data)

var message = "Hello"
take_ownership(message^)  # Transfer ownership
# message is no longer accessible here
```text

**Trait conformance**:

```mojo
struct MyType(Copyable, Movable, Stringable):
    var value: Int

    fn __str__(self) -> String:
        return str(self.value)
```text

#### Quick Reference: Migration Checklist

When reviewing or writing Mojo code:

- [ ] Replace all `inout` with `mut`
- [ ] Replace all `@value` with `@fieldwise_init` + `(Copyable, Movable)`
- [ ] Replace all `DynamicVector` with `List`
- [ ] Replace all `-> (Type1, Type2)` with `-> Tuple[Type1, Type2]`
- [ ] Use `read` (default), `mut`, `var`, or `ref` for parameter conventions
- [ ] Add explicit trait conformances: `(Copyable, Movable)` at minimum

**Reference**: [Mojo Manual - Types](https://docs.modular.com/mojo/manual/types/) | [Value Ownership](https://docs.modular.com/mojo/manual/values/ownership/)

### Mojo Compiler as Source of Truth

**CRITICAL PRINCIPLE**: When there is ANY confusion or uncertainty about Mojo syntax, the Mojo compiler is the **single source of truth**.

#### Why This Matters

Documentation, issues, and even AI-generated code can be incorrect or outdated. The Mojo compiler is authoritative and will reject invalid syntax.

#### Verification Process

When uncertain about Mojo syntax:

1. **Create a minimal test file** with the syntax in question
2. **Run the Mojo compiler** (`mojo build` or `mojo test`)
3. **Trust the compiler output** - if it compiles, the syntax is valid
4. **Document the findings** if they contradict other sources

#### Real-World Example: PR #1982

**The Mistake**:
- Issue #1968 claimed `out self` was "deprecated" in `__init__` methods
- PR #1982 attempted to change FROM `out self` TO `mut self`
- This would have introduced INCORRECT syntax

**The Reality** (confirmed by compiler):
```mojo
# CORRECT for constructors
fn __init__(out self, value: Int):
    self.value = value

# INCORRECT for constructors (would fail compilation)
fn __init__(mut self, value: Int):  # ❌ Wrong!
    self.value = value
```

**Mojo v0.25.7+ Convention**:

- `out self` ✅ - Constructors (`__init__`) that create new instances
- `mut self` - Methods that mutate existing instances
- `read` (default) - Methods that read but don't mutate

**The Fix**:

- Tested with Mojo compiler: `out self` compiles successfully in `__init__`
- Tested with Mojo compiler: `mut self` in `__init__` would fail
- Closed PR #1982 and Issue #1968 as incorrect
- Prevented introducing bugs into the codebase

#### Best Practices

1. **Always verify syntax** with the compiler before making PRs
2. **Never trust documentation alone** - versions may be outdated
3. **Test edge cases** - create minimal reproducible examples
4. **Update documentation** when you find discrepancies
5. **Use compiler errors** as learning opportunities

#### Quick Compiler Verification

```bash
# Create test file
cat > /tmp/test_syntax.mojo << 'EOF'
struct TestStruct:
    var value: Int

    fn __init__(out self, value: Int):
        self.value = value
EOF

# Verify syntax
mojo build /tmp/test_syntax.mojo

# If it compiles → syntax is valid
# If it fails → compiler shows correct syntax
```

**Remember**: The compiler never lies. When in doubt, compile.

### Common Mistakes to Avoid (From 64+ Test Failure Analysis)

**Source**: [Complete Pattern Analysis](docs/dev/mojo-test-failure-patterns.md)

Based on systematic analysis of 10 PRs fixing 64+ test failures, here are the most critical patterns to avoid:

#### 1. Ownership Violations (40% of Failures)

**❌ NEVER**: Pass temporary expressions to functions requiring ownership

```mojo
# WRONG - Cannot transfer ownership of temporary
var labels = ExTensor(List[Int](), DType.int32)
```

**✅ ALWAYS**: Create named variables for ownership transfer

```mojo
# CORRECT - Named variable can be transferred
var labels_shape = List[Int]()
var labels = ExTensor(labels_shape, DType.int32)
```

**❌ NEVER**: Mark structs `ImplicitlyCopyable` when fields contain List/Dict/String

```mojo
# WRONG - List[Float32] is NOT ImplicitlyCopyable
struct SimpleMLP(Copyable, Movable, ImplicitlyCopyable):
    var weights: List[Float32]  # Compiler error!
```

**✅ ALWAYS**: Use explicit transfer operator `^` for non-copyable returns

```mojo
# CORRECT - Explicit ownership transfer
fn get_weights(self) -> List[Float32]:
    return self.weights^
```

#### 2. Constructor Signatures (25% of Failures)

**❌ NEVER**: Use `mut self` in `__init__` constructors

```mojo
# WRONG - Constructors create new instances
fn __init__(mut self, value: Int):
    self.value = value
```

**✅ ALWAYS**: Use `out self` for ALL constructors

```mojo
# CORRECT - out self for constructors
fn __init__(out self, value: Int):
    self.value = value
```

**Constructor Convention Table**:

| Method             | Parameter                  | Example                                          |
| ------------------ | -------------------------- | ------------------------------------------------ |
| `__init__`         | `out self`                 | `fn __init__(out self, value: Int)`              |
| `__moveinit__`     | `out self, deinit existing`| `fn __moveinit__(out self, deinit existing: Self)` |
| `__copyinit__`     | `out self, existing`       | `fn __copyinit__(out self, existing: Self)`      |
| Mutating methods   | `mut self`                 | `fn modify(mut self)`                            |
| Read-only methods  | `read` (implicit)          | `fn get_value(self) -> Int`                      |

#### 3. Uninitialized Data (20% of Failures)

**❌ NEVER**: Assign to list indices without appending first

```mojo
# WRONG - Cannot assign to uninitialized index
var list = List[Int]()
list[0] = 42  # Runtime error - index out of bounds
```

**✅ ALWAYS**: Use `append()` to add new elements

```mojo
# CORRECT - append creates the element
var list = List[Int]()
list.append(42)  # Now list[0] exists
```

**❌ NEVER**: Create ExTensor with empty shape then access multiple indices

```mojo
# WRONG - Empty shape is 0D scalar (1 element only)
var shape = List[Int]()
var tensor = ExTensor(shape, DType.float32)
tensor._data[0] = 1.0
tensor._data[1] = 2.0  # SEGFAULT - out of bounds!
```

**✅ ALWAYS**: Initialize shape dimensions before creating tensors

```mojo
# CORRECT - 1D tensor with 4 elements
var shape = List[Int]()
shape.append(4)
var tensor = ExTensor(shape, DType.float32)
# Now can safely access indices 0-3
```

#### 4. Type System Issues (10% of Failures)

**❌ NEVER**: Use `assert_equal()` for DType comparisons

```mojo
# WRONG - DType doesn't conform to Comparable trait
assert_equal(tensor._dtype, DType.float32)
```

**✅ ALWAYS**: Use `assert_true()` with `==` operator for DType

```mojo
# CORRECT - == works, but needs assert_true
assert_true(tensor._dtype == DType.float32, "Expected float32")
```

**❌ NEVER**: Access methods as properties

```mojo
# WRONG - dtype is a method, not a property
if tensor.dtype == DType.float32:
```

**✅ ALWAYS**: Call methods with parentheses

```mojo
# CORRECT - Call method with ()
if tensor.dtype() == DType.float32:
```

#### 5. Syntax Errors (5% of Failures)

**❌ COMMON TYPO**: Missing space after `var` keyword

```mojo
# WRONG - Typo causing undeclared identifier
vara = ones(shape, DType.float32)
varb = ones(shape, DType.float32)
```

**✅ CORRECT**: Always add space after `var`

```mojo
# CORRECT - Space after var
var a = ones(shape, DType.float32)
var b = ones(shape, DType.float32)
```

#### 6. ExTensor API Patterns (From PR #2168)

**❌ NEVER**: Use static methods on ExTensor - they don't exist

```mojo
# WRONG - ExTensor has no static methods
var tensor = ExTensor.full(shape, 1.0, DType.float32)
var zeros_tensor = ExTensor.zeros(shape, DType.float32)
var value = tensor.item()  # .item() doesn't exist
```

**✅ ALWAYS**: Use standalone functions from shared.core.extensor

```mojo
# CORRECT - Use module-level functions
from shared.core.extensor import full, zeros, ones

var tensor = full(shape, 1.0, DType.float32)
var zeros_tensor = zeros(shape, DType.float32)
var value = tensor._get_float64(0)  # Access first element
```

#### 7. Deprecated Pointer API (Mojo v0.25.7+)

**❌ NEVER**: Use `Pointer.address_of()` - it no longer exists

```mojo
# WRONG - Deprecated API
from memory import Pointer
var ptr = Pointer.address_of(float_val)
var bits = ptr.bitcast[UInt32]()[]
```

**✅ ALWAYS**: Use SIMD bitcast for type punning

```mojo
# CORRECT - SIMD bitcast
from memory import bitcast
var bits = bitcast[DType.uint32, 1](SIMD[DType.float32, 1](float_val))[0]
```

#### 8. Type Casting Syntax

**❌ NEVER**: Use old `.cast[DType]()` syntax

```mojo
# WRONG - Old syntax
var bits = value.cast[DType.uint8]()
var text = str(dtype)
```

**✅ ALWAYS**: Use constructor-style casts

```mojo
# CORRECT - Constructor syntax
var bits = UInt8(value)
var text = String(dtype)
```

#### 9. Binary Operations with Tensors

**❌ NEVER**: Pass literals to tensor binary operations

```mojo
# WRONG - power() requires two tensors
var squared = power(tensor, 2.0)
```

**✅ ALWAYS**: Create tensor for second operand

```mojo
# CORRECT - Use full_like to create matching tensor
var exponent = full_like(tensor, 2.0)
var squared = power(tensor, exponent)
```

#### 10. Test Assertion Patterns

**❌ NEVER**: Pass Float32 directly to assert_almost_equal

```mojo
# WRONG - Float32 may not match expected signature
assert_almost_equal(tensor._data.bitcast[Float32]()[0], 0.99, tolerance=1e-6)
```

**✅ ALWAYS**: Wrap Float32 in Float64 for assertions

```mojo
# CORRECT - Explicit Float64 conversion
assert_almost_equal(Float64(tensor._data.bitcast[Float32]()[0]), 0.99, tolerance=1e-6)
```

**Numerical Gradient Tolerances**: Float32 gradients need relaxed tolerances

```mojo
# For float32 numerical gradient checking:
# - Use rtol=1e-2, atol=1e-2 for Conv2D operations
# - Use rtol=1e-3, atol=1e-3 for cross-entropy loss
# - Use rtol=1e-3, atol=5e-4 for softmax operations
```

#### 11. List Initialization

**Per [Mojo Manual - List](https://docs.modular.com/mojo/manual/types#list)**: Use list literals for initialization.

```mojo
# CORRECT - List literal (type inference)
var shape = [3, 4, 5]  # Inferred as List[Int]

# CORRECT - Explicit type annotation
var shape: List[Int] = [3, 4, 5]

# CORRECT - Empty list requires explicit type
var empty = List[Int]()

# ❌ WRONG - Variadic constructor does not exist
var shape = List[Int](3, 4, 5)  # Compiler error: no matching function
```

### Critical Pre-Flight Checklist

Before committing Mojo code, verify:

**Ownership & Initialization:**

- [ ] All `__init__` methods use `out self` (not `mut self`)
- [ ] All List/Dict/String returns use `^` transfer operator
- [ ] All List operations use `append()` for new elements (not `list[i] = value` on empty list)
- [ ] All ExTensor shapes initialized with dimensions (not empty `List[Int]()` for multi-element access)
- [ ] All test tensors have ALL elements initialized (check `numel()`)
- [ ] No `ImplicitlyCopyable` trait on structs with List/Dict/String fields
- [ ] No temporary expressions passed to `var` parameters

**API Usage:**

- [ ] Use `full()`, `zeros()`, `ones()` functions (NOT `ExTensor.full()`, etc.)
- [ ] Use `._get_float64(0)` for scalar access (NOT `.item()`)
- [ ] Use `bitcast[]` for type punning (NOT `Pointer.address_of()`)
- [ ] Use `UInt8(value)` for casts (NOT `value.cast[DType.uint8]()`)
- [ ] Use `String(dtype)` (NOT `str(dtype)`)
- [ ] Binary tensor ops use two tensors (NOT `power(tensor, 2.0)`)

**Type System:**

- [ ] DType comparisons use `assert_true(a == b)` not `assert_equal(a, b)`
- [ ] Methods called with `()`: `tensor.dtype()` not `tensor.dtype`
- [ ] Float32 values wrapped in `Float64()` for `assert_almost_equal()`
- [ ] Closures use `escaping` keyword when captured by other functions

**Code Quality:**

- [ ] All package functions exported in `__init__.mojo`
- [ ] Space after `var` keyword: `var a` not `vara`
- [ ] Verify function signatures before calling (check source or compiler)

**See**: [Complete Mojo Failure Patterns](docs/dev/mojo-test-failure-patterns.md) for detailed
examples and prevention strategies.

### Zero-Warnings Policy

**CRITICAL**: This project enforces a zero-warnings policy for ALL Mojo code to maintain code quality and catch
potential bugs early.

#### Why We Enforce This

- **Catch bugs early**: Many warnings indicate potential runtime errors or logic bugs
- **Prevent warning accumulation**: Small warnings compound over time and become harder to fix
- **Enforce consistency**: Ensures all code follows the same quality standards
- **Make failures explicit**: Code with warnings will not be merged

#### How It Works

**Note**: Mojo doesn't support a `-Werror` flag (as of v0.25.7). Instead, we enforce zero warnings through:

1. **Code review**: PRs with warnings are rejected
2. **CI monitoring**: Warnings are visible in CI logs and must be addressed
3. **Developer discipline**: Fix warnings immediately, don't accumulate them

**pixi.toml tasks**: Standard Mojo commands (warnings visible in output)

```toml
[tasks]
build = "mojo build"  # Warnings must be fixed, not suppressed
run = "mojo run"      # Warnings must be fixed, not suppressed
test = "mojo test"    # Warnings must be fixed, not suppressed
format = "mojo format"
```

#### Common Warning Patterns to Avoid

**1. Unused Variables**

```mojo
# ❌ WRONG - Unused loop variable
for i in range(10):
    list.append(0)

# ✅ CORRECT - Use underscore for unused variables
for _ in range(10):
    list.append(0)
```

**2. Unused Function Parameters**

```mojo
# ❌ WRONG - Unused parameter
fn process(data: ExTensor, unused_param: Int):
    return data

# ✅ CORRECT - Remove unused parameter or prefix with underscore
fn process(data: ExTensor, _debug_level: Int):
    return data
```

**3. Mutating Method on Immutable Reference**

```mojo
# ❌ WRONG - Calling mutating method on read-only reference
fn iterate(loader: BatchLoader):
    var batches = loader.__iter__()  # Error: __iter__ needs mut self

# ✅ CORRECT - Use mutable reference
fn iterate(mut loader: BatchLoader):
    var batches = loader.__iter__()
```

**4. Missing Transfer Operator for Non-Copyable Types**

```mojo
# ❌ WRONG - List/Dict/String fields need transfer operator
fn get_strides(self) -> List[Int]:
    return self._strides  # Warning: implicit copy of non-copyable type

# ✅ CORRECT - Use transfer operator
fn get_strides(self) -> List[Int]:
    return self._strides^
```

#### Verification

Before committing, verify your code compiles without warnings:

```bash
# Test individual file
pixi run mojo -I . tests/shared/core/test_example.mojo

# Build and check for warnings in output
pixi run mojo build -I . shared/core/extensor.mojo

# Run all tests and verify no warnings
pixi run mojo test -I . tests/
```

**Check the output carefully**: If you see any warnings, fix them before committing.

#### When You See a Warning

1. **Read the warning message carefully** - Mojo warnings are specific and actionable
2. **Fix the root cause** - Don't suppress or work around warnings
3. **Test the fix** - Verify the warning is gone and code still works
4. **Document if unusual** - Add comments if the fix is non-obvious

**Remember**: If the Mojo compiler warns about it, there's usually a good reason. Fix it, don't ignore it.

## Environment Setup

This project uses Pixi for environment management:

```bash

# Pixi is already configured - dependencies are in pixi.toml
# Mojo is the primary language target for future implementations

```text

## Common Commands

### Justfile Build System

The project uses [Just](https://just.systems/) as a unified command runner for local development and CI/CD consistency.

#### Quick Reference

```bash
# Show all available recipes
just --list

# Get help
just help

# Development commands
just build                  # Build project in debug mode
just test                   # Run all tests
just test-mojo             # Run only Mojo tests
just lint                  # Run all linters
just format                # Format all files

# CI-specific commands (match GitHub Actions)
just ci-validate           # Full CI validation (build + test)
just ci-build              # Build shared package
just ci-compile            # Compile package (validation only)
just ci-test-mojo          # Run all Mojo tests
just ci-test-group PATH PATTERN  # Run specific test group
just ci-lint               # Run pre-commit hooks

# Training and inference
just train                 # Train LeNet-5 with defaults
just train lenet5 fp16 20  # Train with FP16, 20 epochs
just infer lenet5 ./weights  # Run inference

# Docker management
just docker-up             # Start development environment
just docker-down           # Stop environment
just docker-shell          # Open shell in container
```

### Why Use Justfile?

1. **Consistency**: Same commands work locally and in CI
2. **Simplicity**: Easy-to-read recipes vs complex bash scripts
3. **Documentation**: Self-documenting with `just --list`
4. **Reliability**: Ensures identical flags between local dev and CI

### CI Integration

GitHub Actions workflows use justfile recipes to ensure consistency:

```yaml
# Example from comprehensive-tests.yml
- name: Run test group
  run: just ci-test-group "tests/shared/core" "test_*.mojo"

# Example from build-validation.yml
- name: Build package
  run: just ci-build
```

This ensures developers can run `just ci-validate` locally to reproduce CI results exactly.

**See**: `justfile` for complete recipe list and implementation details.

### GitHub CLI

```bash

# Check authentication status

gh auth status

# List issues

gh issue list

# View issue details

gh issue view <number>

# Reply to PR review comments (addressing feedback)

gh pr comment <pr-number> --body "Short, concise explanation of what was done"
```text

### Handling PR Review Comments

**CRITICAL**: There are TWO types of comments - do NOT confuse them:

1. **PR-level comments** - General comments in the PR timeline (`gh pr comment`)
1. **Review comment replies** - Specific replies to inline code review comments (GitHub API)

When addressing review comments on a pull request:

1. **Make the requested changes** in your code
1. **Reply to EACH review comment individually** using the correct API
1. **Verify replies were posted** before reporting completion
1. **Check CI status** after pushing changes

#### Correct Way to Reply to Review Comments

**DO NOT USE** `gh pr comment` - that creates a general PR comment, not a reply to review comments.

### CORRECT approach:

```bash
# Step 1: Get review comment IDs
gh api repos/OWNER/REPO/pulls/PR/comments --jq '.[] | select(.user.login == "REVIEWER") | {id: .id, path: .path, body: .body}'

# Step 2: Reply to EACH comment
gh api repos/OWNER/REPO/pulls/PR/comments/COMMENT_ID/replies \
  --method POST \
  -f body="✅ Fixed - [brief description]"

# Step 3: Verify replies posted
gh api repos/OWNER/REPO/pulls/PR/comments --jq '.[] | select(.in_reply_to_id)'

# Step 4: Check CI status
sleep 30  # Wait for CI to start
gh pr checks PR
```text

### Example responses

- `✅ Fixed - Updated conftest.py to use real repository root instead of mock tmp_path`
- `✅ Fixed - Deleted test_link_validation.py since link validation is handled by pre-commit`
- `✅ Fixed - Removed markdown linting section from README.md`

### Important

- Keep responses SHORT and CONCISE (1 line preferred)
- Start with ✅ to indicate the issue is resolved
- Explain WHAT was done, not why (unless asked)
- Reply to ALL open review comments individually
- **VERIFY** replies were posted - don't assume
- **CHECK CI** status after pushing - local pre-commit can differ from CI

**See detailed guide:** `/agents/guides/github-review-comments.md`

**See verification checklist:** `/agents/guides/verification-checklist.md`

### Agent Testing

Agent configurations are automatically validated in CI on all PRs. Run tests locally before committing:

```bash
# Validate agent YAML frontmatter and configuration
python3 tests/agents/validate_configs.py .claude/agents/

# Test agent discovery and loading
python3 tests/agents/test_loading.py .claude/agents/

# Test delegation patterns
python3 tests/agents/test_delegation.py .claude/agents/

# Test workflow integration
python3 tests/agents/test_integration.py .claude/agents/

# Test Mojo-specific patterns
python3 tests/agents/test_mojo_patterns.py .claude/agents/

# Run all tests
for script in tests/agents/test_*.py tests/agents/validate_*.py; do
    python3 "$script" .claude/agents/
done
```text

### Test Coverage

- Configuration validation (YAML frontmatter, required fields, tool specifications)
- Agent discovery and loading (hierarchy coverage, activation patterns)
- Delegation patterns (chain validation, escalation paths)
- Workflow integration (5-phase coverage, parallel execution)
- Mojo patterns (fn vs def, struct vs class, SIMD, memory management)

**CI Integration**: The `.github/workflows/test-agents.yml` workflow runs these tests automatically on all PRs
affecting agent configurations.

### Pre-commit Hooks

Pre-commit hooks automatically check code quality before commits. The hooks include `mojo format` for Mojo code and
markdown linting for documentation.

```bash

# Install pre-commit hooks (one-time setup)

pre-commit install

# Run hooks manually on all files

pre-commit run --all-files

# Run hooks manually on staged files only

pre-commit run

# Skip hooks (use sparingly, only when necessary)

git commit --no-verify
```text

### Configured Hooks

- `mojo format` - Auto-format Mojo code (`.mojo`, `.🔥` files)
- `markdownlint-cli2` - Lint markdown files (currently disabled, will enable after fixing existing files)
- `trailing-whitespace` - Remove trailing whitespace
- `end-of-file-fixer` - Ensure files end with newline
- `check-yaml` - Validate YAML syntax
- `check-added-large-files` - Prevent large files from being committed (max 1MB)
- `mixed-line-ending` - Fix mixed line endings

**CI Enforcement**: The `.github/workflows/pre-commit.yml` workflow runs these checks on all PRs and pushes to `main`.

## Repository Architecture

### Project Structure

```text
ml-odyssey/
├── agents/                      # Team documentation
│   ├── README.md                # Quick start guide
│   ├── hierarchy.md             # Visual hierarchy diagram
│   ├── agent-hierarchy.md       # Complete agent specifications
│   ├── delegation-rules.md      # Coordination patterns
│   └── templates/               # Agent configuration templates
├── notes/
│   └── review/                  # Comprehensive specs & architectural decisions
│       ├── agent-architecture-review.md
│       ├── skills-design.md
│       └── orchestration-patterns.md
├── scripts/                     # Python automation scripts
├── logs/                        # Execution logs and state files
└── .clinerules                 # Claude Code conventions
```text

### Planning Hierarchy

**4 Levels** (managed through GitHub issues):

1. **Section** (e.g., 01-foundation) - Major area of work
1. **Subsection** (e.g., 01-directory-structure) - Logical grouping
1. **Component** (e.g., 01-create-papers-dir) - Specific deliverable
1. **Subcomponent** (e.g., 01-create-base-dir) - Atomic task

All planning documentation is tracked in GitHub issues. Use `gh issue view <number>` to read plans.

### Documentation Organization

The repository uses three separate locations for documentation to avoid duplication:

#### 1. Team Documentation (`/agents/`)

**Purpose**: Quick start guides, visual references, and templates for team onboarding.

### Contents

- Quick start guides (README.md)
- Visual diagrams (hierarchy.md)
- Quick reference cards (delegation-rules.md)
- Configuration templates (templates/)

**When to Use**: Creating new documentation for team onboarding or quick reference.

#### 2. Developer Documentation (`/docs/dev/`)

**Purpose**: Detailed architectural decisions, comprehensive specifications, and design documents.

### Contents

- Mojo patterns and error handling (mojo-test-failure-patterns.md)
- Skills architecture (skills-design.md, skills-architecture.md)
- Orchestration patterns (orchestration-patterns.md)
- Backward pass catalog (backward-pass-catalog.md)

**When to Use**: Writing detailed specifications, architectural decisions, or comprehensive guides.

#### 3. Issue-Specific Documentation (GitHub Issue Comments)

**Purpose**: Implementation notes, findings, and decisions specific to a single GitHub issue.

**Location**: Post directly to the GitHub issue as comments using `gh issue comment`.

**Reading Issue Context**:

```bash
# Get issue details and body
gh issue view <number>

# Get all comments (implementation history)
gh issue view <number> --comments

# Get structured data
gh issue view <number> --json title,body,comments,labels,state
```

**Writing to Issues**:

```bash
# Post implementation notes
gh issue comment <number> --body "$(cat <<'EOF'
## Implementation Notes

### Summary
[What was implemented]

### Files Changed
- path/to/file.mojo

### Verification
- [x] Tests pass
EOF
)"
```

### Important Rules

- ✅ DO: Post issue-specific findings and decisions as comments
- ✅ DO: Link to comprehensive docs in `/agents/` and `/docs/dev/`
- ✅ DO: Reference related issues with `#<number>` format
- ❌ DON'T: Duplicate comprehensive documentation
- ❌ DON'T: Create local files for issue tracking

### 5-Phase Development Workflow

Every component follows a hierarchical workflow with clear dependencies:

**Workflow**: Plan → [Test | Implementation | Package] → Cleanup

1. **Plan** - Design and documentation (MUST complete first)
1. **Test** - Write tests following TDD (parallel after Plan)
1. **Implementation** - Build the functionality (parallel after Plan)
1. **Package** - Create distributable packages (parallel after Plan)
   - Build binary packages (`.mojopkg` files for Mojo modules)
   - Create distribution archives (`.tar.gz`, `.zip` for tooling/docs)
   - Configure package metadata and installation procedures
   - Add components to existing packages
   - Test package installation in clean environments
   - Create CI/CD packaging workflows
   - **NOT just documenting** - must create actual distributable artifacts
1. **Cleanup** - Refactor and finalize (runs after parallel phases complete)

### Key Points

- Plan phase produces specifications for all other phases
- Test/Implementation/Package can run in parallel after Plan completes
- Cleanup collects issues discovered during the parallel phases
- Each phase has a separate GitHub issue with detailed instructions

## GitHub Issue Structure

All planning is done through GitHub issues with clear structure:

### Issue Body Format

```markdown
## Objective
Brief description (2-3 sentences)

## Deliverables
- [ ] Deliverable 1
- [ ] Deliverable 2

## Success Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Dependencies
- Depends on #<parent-issue>
- Related: #<sibling-issue>

## Notes
Additional context
```

### Issue Labels

- `planning` - Design phase
- `testing` - Test development
- `implementation` - Code implementation
- `packaging` - Distribution packages
- `cleanup` - Finalization

### Linking Issues

- Reference in body: `Depends on #123`
- Reference in commits: `Implements #123`
- Close via PR: `Closes #123`

## Working with GitHub Issues

All planning and documentation is managed through GitHub issues directly.

### Creating New Work Items

1. Create a GitHub issue with clear description and acceptance criteria
2. Use appropriate labels (planning, testing, implementation, packaging, cleanup)
3. Link related issues using `#<number>` references

### Tracking Implementation

1. Read issue context: `gh issue view <number> --comments`
2. Post progress updates as issue comments
3. Link PRs to issues: `gh pr create --body "Closes #<number>"`

### Documentation Workflow

1. **Read context first**: `gh issue view <number> --comments`
2. **Post updates**: `gh issue comment <number> --body "..."`
3. **Reference in commits**: "Implements #<number>" or "Closes #<number>"

See `.claude/shared/github-issue-workflow.md` for complete workflow patterns.

### File Locations

- **Scripts**: `scripts/*.py`
- **Logs**: `logs/*.log`
- **Tracked Docs**: `docs/dev/`, `agents/` (reference these in commits)
- **Issue Docs**: GitHub issue comments (not local files)

## Git Workflow

### Branch Naming

- `main` - Production branch (protected, requires PR)
- `<issue-number>-<description>` - Feature/fix branches (e.g., `1928-consolidate-test-assertions`)

### Development Workflow

**IMPORTANT:** The `main` branch is protected. All changes must go through a pull request.

#### Creating a PR (Standard Workflow)

1. **Create a feature branch:**

   ```bash
   git checkout -b <issue-number>-<description>
   ```

1. **Make your changes and commit:**

   ```bash
   git add <files>
   git commit -m "$(cat <<'EOF'
   type(scope): Brief description

   Detailed explanation of changes.

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>
   EOF
   )"
   ```

1. **Push the feature branch:**

   ```bash
   git push -u origin <branch-name>
   ```

1. **Create pull request:**

   ```bash
   gh pr create \
     --title "[Type] Brief description" \
     --body "Closes #<issue-number>" \
     --label "appropriate-label"
   ```

1. **Enable auto-merge:**

   ```bash
   gh pr merge --auto --rebase
   ```

   **Always enable auto-merge** so PRs merge automatically once CI passes.

### Never Push Directly to Main

❌ **NEVER DO THIS:**

```bash
git checkout main
git commit -m "changes"
git push origin main  # Will be rejected - main is protected
```

✅ **ALWAYS DO THIS:**

```bash
git checkout -b <issue-number>-description
git commit -m "changes"
git push -u origin <issue-number>-description
gh pr create --title "..." --body "Closes #<issue>" --label "..."
gh pr merge --auto --rebase  # Enable auto-merge
```

### Commit Message Format

Follow conventional commits:

```text
feat(section): Add new component
fix(scripts): Correct parsing issue
docs(readme): Update instructions
refactor(plans): Standardize to Template 1
```text

### Worktree and PR Discipline

**One PR per Issue:**

- Each GitHub issue should have exactly ONE pull request
- Do not combine multiple issues into a single PR
- Branch naming: `<issue-number>-<description>`

**Worktree Directory:**

- Create all worktrees in the `worktrees/` subdirectory within the repo
- Naming convention: `<issue-number>-<description>`
- Example: `git worktree add worktrees/123-fix-bug 123-fix-bug`

**Post-Merge Cleanup:**

After a PR is merged/rebased to main:

1. Remove the worktree: `git worktree remove worktrees/<issue-number>-<description>`
2. Delete local branch: `git branch -d <branch-name>`
3. Delete remote branch: `git push origin --delete <branch-name>`
4. Prune stale references: `git worktree prune`

## Labels

Standard labels automatically created by scripts:

- `planning` - Design phase (light purple: #d4c5f9)
- `documentation` - Documentation work (blue: #0075ca)
- `testing` - Testing phase (yellow: #fbca04)
- `tdd` - Test-driven development (yellow: #fbca04)
- `implementation` - Implementation phase (dark blue: #1d76db)
- `packaging` - Integration/packaging (light green: #c2e0c6)
- `integration` - Integration tasks (light green: #c2e0c6)
- `cleanup` - Cleanup/finalization (red: #d93f0b)

## Python Coding Standards

```python

#!/usr/bin/env python3

"""
Script description

Usage:
    python scripts/script_name.py [options]
"""

# Standard imports first

import sys
import re
from pathlib import Path
from typing import List, Dict, Optional

def function_name(param: str) -> bool:
    """Clear docstring with purpose, params, returns."""
    pass
```text

### Requirements

- Python 3.7+
- Type hints required for all functions
- Clear docstrings for public functions
- Comprehensive error handling
- Logging for important operations

## Markdown Standards

All markdown files must follow these standards to pass `markdownlint-cli2` linting:

### Code Blocks (MD031, MD040)

**Rule**: Fenced code blocks must be:

1. Surrounded by blank lines (before and after)
1. Have a language specified

### Correct

```markdown

Some text before.

```python

def hello():
    print("world")

```text
Some text after.

```text

### Incorrect

```markdown
Some text before.
```text

def hello():

```text
Some text after.
```text

### Language Examples

- Python: ` ```python `
- Bash: ` ```bash `
- Text/plain: ` ```text `
- Mojo: ` ```mojo `
- YAML: ` ```yaml `
- JSON: ` ```json `
- Markdown: ` ```markdown `

### Lists (MD032)

**Rule**: Lists must be surrounded by blank lines (before and after)

### Correct

```markdown
Some text before.

- Item 1
- Item 2
- Item 3

Some text after.
```text

### Incorrect

```markdown
Some text before.
- Item 1
- Item 2
Some text after.
```text

### Headings (MD022)

**Rule**: Headings must be surrounded by blank lines (one blank line before and after)

### Correct

```markdown
Some content here.

## Section Heading

More content here.
```text

### Incorrect

```markdown
Some content here.
## Section Heading
More content here.
```text

### Line Length (MD013)

**Rule**: Lines should not exceed 120 characters (except for URLs or code blocks)

### Guidelines

- Break long lines at 120 characters
- For long sentences, break at natural boundaries (clauses, lists, etc.)
- Code in code blocks is exempt
- URLs in links are exempt (use reference-style links if needed)

### Example

```markdown
This is a very long sentence that exceeds the 120 character limit and should be broken into
multiple lines at a natural boundary point for better readability.
```text

### Best Practices

1. **Always add blank lines around code blocks and lists** - This is the #1 cause of linting failures
1. **Always specify language for code blocks** - Use appropriate language tags
1. **Check headings have surrounding blank lines** - Especially after subheadings
1. **Use reference-style links for long URLs** - Helps avoid line length issues

### Quick Checklist for New Content

Before committing markdown files:

- [ ] All code blocks have a language specified (` ```python ` not ` ``` `)
- [ ] All code blocks have blank lines before and after
- [ ] All lists have blank lines before and after
- [ ] All headings have blank lines before and after
- [ ] No lines exceed 120 characters
- [ ] File ends with newline (enforced by pre-commit)
- [ ] No trailing whitespace (enforced by pre-commit)

### Running Markdown Linting Locally

```bash

# Check specific file
npx markdownlint-cli2 path/to/file.md

# Check all markdown files
pre-commit run markdownlint-cli2 --all-files

# View detailed errors
npx markdownlint-cli2 path/to/file.md 2>&1

```text

## Debugging

### Check Logs

```bash
# View script logs
tail -100 logs/*.log

# View specific log
cat logs/<script>_*.log
```text

## Troubleshooting

### GitHub CLI Issues

```bash
# Check authentication
gh auth status

# If missing scopes, refresh authentication
gh auth refresh -h github.com
```text

### Issue Access Problems

- Check GitHub CLI auth: `gh auth status`
- Verify repository access: `gh repo view`
- Test issue access: `gh issue list`

### Script Errors

- Verify Python version: `python3 --version` (requires 3.7+)
- Check file permissions
- Review error logs in `logs/` directory

## Important Files

- `.clinerules` - Comprehensive Claude Code conventions
- `docs/dev/` - Developer documentation (Mojo patterns, skills architecture)
- `scripts/README.md` - Complete scripts documentation
- `README.md` - Main project documentation
- `.claude/shared/github-issue-workflow.md` - GitHub issue read/write patterns
