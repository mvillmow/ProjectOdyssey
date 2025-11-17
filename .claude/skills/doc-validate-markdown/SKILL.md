---
name: doc-validate-markdown
description: Validate markdown files for formatting, links, and style compliance using markdownlint. Use before committing documentation changes.
---

# Validate Markdown Skill

Validate markdown files for formatting and style.

## When to Use

- Before committing documentation
- Markdown linting errors in CI
- Creating new documentation
- Updating existing docs

## Usage

```bash
# Validate all markdown
npx markdownlint-cli2 "**/*.md"

# Validate specific file
npx markdownlint-cli2 README.md

# Fix auto-fixable issues
npx markdownlint-cli2 --fix "**/*.md"
```

## Common Issues

### MD040: Code blocks need language

````markdown
# ❌ Wrong
```
code here
```

# ✅ Correct
```python
code here
```
````

### MD031: Blank lines around code blocks

````markdown
# ❌ Wrong
Text before
```python
code
```
Text after

# ✅ Correct

Text before

```python
code
```

Text after
````

### MD013: Line too long

Break lines at 120 characters.

## Configuration

`.markdownlint.yaml`:

```yaml
line-length:
  line_length: 120
  code_blocks: false
  tables: false
```

See `quality-run-linters` for complete linting workflow.
