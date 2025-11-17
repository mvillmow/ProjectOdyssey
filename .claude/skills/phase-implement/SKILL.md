---
name: phase-implement
description: Coordinate implementation phase by delegating tasks to engineers, monitoring progress, and ensuring code quality. Use during the implementation phase of the 5-phase development workflow.
---

# Implementation Phase Coordination Skill

This skill coordinates the implementation phase by delegating tasks and ensuring quality.

## When to Use

- User asks to coordinate implementation (e.g., "coordinate implementation phase")
- Implementation phase of 5-phase workflow
- Need to delegate implementation tasks
- Managing multiple implementation tasks

## 5-Phase Workflow Context

**Workflow**: Plan → [Test | Implementation | Package] → Cleanup

Implementation phase runs in parallel with Test and Package phases after Plan completes.

## Coordination Workflow

### 1. Review Plan Specifications

```bash
# Read plan documentation
cat notes/plan/<section>/<subsection>/plan.md

# Review success criteria
grep "Success Criteria" -A 10 plan.md
```

### 2. Break Down Implementation

```bash
# Generate implementation tasks
./scripts/create_implementation_tasks.sh <component-name>

# This creates:
# - Task list with priorities
# - Delegation assignments
# - Dependencies between tasks
```

### 3. Delegate to Engineers

Delegate based on complexity:

**Complex** → Senior Implementation Engineer

- Algorithms
- Performance-critical code
- SIMD optimizations

**Standard** → Implementation Engineer

- Standard functions
- Business logic
- Data structures

**Simple** → Junior Implementation Engineer

- Boilerplate
- Simple helpers
- Type definitions

### 4. Monitor Progress

```bash
# Check implementation status
./scripts/check_implementation_status.sh

# Shows:
# - Completed tasks
# - In-progress tasks
# - Blocked tasks
# - Quality metrics
```

### 5. Code Review

Review all implementations for:

- **Quality** - Clean, maintainable code
- **Standards** - Follows Mojo guidelines
- **Tests** - Adequate coverage
- **Documentation** - Clear comments
- **Performance** - Meets requirements

## Implementation Standards

### Mojo-Specific

#### Function Definitions

```mojo
# Use fn for performance-critical code
fn add_vectors[dtype: DType](
    a: Tensor[dtype],
    b: Tensor[dtype]
) -> Tensor[dtype]:
    """Add two tensors element-wise with SIMD optimization."""
    # Implementation
    pass

# Use def for flexibility
def helper_function(data: String):
    # Implementation
    pass
```

#### Memory Management

```mojo
# Use owned for ownership transfer
fn process(owned data: Tensor) -> Tensor:
    return data^  # Move ownership

# Use borrowed for read-only access
fn read_only(borrowed data: Tensor):
    # Can't modify data
    pass

# Use inout for mutable references
fn modify(inout data: Tensor):
    # Can modify data
    pass
```

#### SIMD Optimization

```mojo
from sys.info import simdwidthof

fn simd_add[dtype: DType](a: Tensor[dtype], b: Tensor[dtype]):
    """SIMD-optimized element-wise addition."""
    alias simd_width = simdwidthof[dtype]()
    # Use SIMD operations
    pass
```

## Delegation Examples

### Example 1: Tensor Operations

```markdown
**Component**: Tensor Operations
**Complexity**: High

**Delegation**:
- Tensor struct → Senior Engineer (complex ownership)
- add function → Engineer (standard SIMD)
- multiply function → Engineer (standard SIMD)
- matmul function → Senior Engineer (complex algorithm)
- Type aliases → Junior Engineer (boilerplate)
```

### Example 2: Data Loaders

```markdown
**Component**: Data Loaders
**Complexity**: Medium

**Delegation**:
- Loader interface → Engineer (standard)
- File reading → Engineer (IO operations)
- Preprocessing → Senior Engineer (performance-critical)
- Batching → Engineer (standard logic)
```

## Quality Checks

Before marking implementation complete:

```bash
# Format code
mojo format src/**/*.mojo

# Run tests
mojo test tests/

# Check coverage
./scripts/check_coverage.sh

# Run linters
./scripts/run_linters.sh

# Verify performance
./scripts/benchmark.sh
```

## Error Handling

### Implementation Blockers

- **Unclear requirements**: Escalate to design agent
- **Performance issues**: Consult performance specialist
- **Test failures**: Coordinate with test specialist
- **Missing dependencies**: Update plan and communicate

### Quality Issues

- **Code smells**: Require refactoring before merge
- **No tests**: Reject until tests added
- **Poor documentation**: Require improvement
- **Performance regression**: Investigate and fix

## Examples

**Start implementation phase:**

```bash
./scripts/start_implementation.sh tensor-operations
```

**Delegate task:**

```bash
./scripts/delegate_task.sh "implement matmul" senior-engineer
```

**Check status:**

```bash
./scripts/check_implementation_status.sh
```

## Scripts Available

- `scripts/create_implementation_tasks.sh` - Break down into tasks
- `scripts/delegate_task.sh` - Assign task to engineer
- `scripts/check_implementation_status.sh` - Monitor progress
- `scripts/review_implementation.sh` - Code quality review

## Integration with Other Phases

- **After Plan** - Receives specifications
- **Parallel with Test** - Coordinates TDD
- **Parallel with Package** - Provides modules for packaging
- **Before Cleanup** - Completes before cleanup starts

## Success Criteria

- [ ] All implementation tasks completed
- [ ] Code quality meets standards
- [ ] Tests passing
- [ ] Performance requirements met
- [ ] Documentation complete
- [ ] Code reviewed and approved

See CLAUDE.md for complete 5-phase workflow documentation.
