---
name: mojo-lint-syntax
description: "Validate Mojo syntax against current v0.25.7+ standards. Use to catch syntax errors before compilation."
category: mojo
mcp_fallback: none
---

# Lint Mojo Syntax

Validate Mojo code against v0.25.7+ syntax standards.

## When to Use

- Writing new Mojo code before testing
- Reviewing Mojo code for syntax issues
- Migrating code from older Mojo versions
- Checking for deprecated patterns
- Pre-commit validation of Mojo files

## Quick Reference

```bash
# Validate single file
mojo build -I . file.mojo

# Check all Mojo files
find . -name "*.mojo" -o -name "*.🔥" | while read f; do
  echo "Checking $f"
  mojo build -I . "$f" 2>&1 | grep -i "error"
done

# Format code (fixes many syntax issues)
pixi run mojo format .

# Check for deprecated patterns
grep -r "inout self\|@value\|DynamicVector\|->" *.mojo | grep -v "result\|fn"
```

## Common Syntax Issues

**Deprecated Patterns**:

- ❌ `inout self` → ✅ `out self` in `__init__`, ✅ `mut self` in methods
- ❌ `@value` → ✅ `@fieldwise_init` with trait list
- ❌ `DynamicVector` → ✅ `List`
- ❌ `-> (T1, T2)` → ✅ `-> Tuple[T1, T2]`

**Constructor Issues**:

- Wrong parameter type in `__init__` (must be `out self`)
- Missing trait conformances (`Copyable`, `Movable`)
- Incorrect initialization order

**Type Issues**:

- Missing type annotations (required in fn declarations)
- Mismatched types in assignments
- Invalid type parameters

**Ownership Issues**:

- Missing transfer operator `^` for non-copyable types
- Using `var` parameter incorrectly
- Copy/move semantics violations

## Validation Workflow

1. **Check syntax**: Run mojo compiler on files
2. **Fix format**: Run `mojo format` to auto-fix style
3. **Verify patterns**: Check for deprecated patterns
4. **Type check**: Ensure all types are correct
5. **Ownership check**: Verify ownership semantics
6. **Compile test**: Build to catch runtime issues
7. **Report issues**: List all problems found

## Output Format

Report syntax issues with:

1. **File** - Which file has the issue
2. **Line** - Line number of error
3. **Error** - Syntax error message
4. **Pattern** - What deprecated/wrong pattern was used
5. **Fix** - How to correct it
6. **Severity** - Critical (won't compile) or warning

## Error Handling

| Problem | Solution |
|---------|----------|
| Compiler not found | Verify mojo is installed and in PATH |
| Module not found | Add `-I .` flag to include current directory |
| Encoding issues | Convert file to UTF-8 |
| Version mismatch | Check mojo version against v0.25.7+ |
| Large files | Process one file at a time |

## Validation Checklist

Before committing Mojo code:

- [ ] File compiles with `mojo build`
- [ ] No syntax errors in compiler output
- [ ] No deprecated patterns (inout, @value, DynamicVector)
- [ ] All `__init__` use `out self` (not `mut self`)
- [ ] All non-copyable returns use `^` operator
- [ ] All type annotations present in fn declarations
- [ ] Zero compiler warnings

## References

- See CLAUDE.md for v0.25.7+ syntax standards
- See validate-mojo-patterns for pattern validation
- See mojo-format skill for code formatting
