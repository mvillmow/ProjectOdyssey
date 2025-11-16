---
name: quality-complexity-check
description: Analyze code complexity metrics including cyclomatic complexity and nesting depth. Use to identify code that needs refactoring.
---

# Complexity Check Skill

Analyze and report code complexity metrics.

## When to Use

- Code review process
- Identifying refactoring candidates
- Maintaining code quality
- Before major releases

## Complexity Metrics

### 1. Cyclomatic Complexity

Measures decision points (if, for, while):
- **1-10**: Simple, easy to test
- **11-20**: Moderate, consider refactoring
- **21+**: Complex, needs refactoring

### 2. Nesting Depth

Maximum levels of nesting:
- **1-3**: Good
- **4-5**: Consider flattening
- **6+**: Needs refactoring

### 3. Function Length

Lines of code in function:
- **1-20**: Good
- **21-50**: Acceptable
- **51+**: Consider splitting

## Usage

```bash
# Analyze Python code
./scripts/check_complexity.py

# Analyze Mojo code (when tools available)
./scripts/check_mojo_complexity.sh

# Generate report
./scripts/complexity_report.sh > complexity.txt
```

## Refactoring Strategies

### High Complexity

```python
# ❌ Complex (CC: 15)
def process(data):
    if condition1:
        if condition2:
            if condition3:
                # nested logic
                for item in data:
                    if item.valid:
                        # more nesting
                        pass

# ✅ Refactored (CC: 5)
def process(data):
    if not is_valid(data):
        return
    filtered = filter_valid_items(data)
    return process_items(filtered)
```

### Deep Nesting

Extract functions to reduce nesting depth.

See `phase-cleanup` for refactoring guidelines.
