---
name: skill-template
description: "Template for new skills following optimized format. Use as reference when creating new skills."
category: example
mcp_fallback: example
---

# Skill Name

One-sentence description of what this skill does.

## When to Use

- Bullet point trigger condition 1
- Bullet point trigger condition 2
- Bullet point trigger condition 3

## Quick Reference

The most critical command or workflow for this skill:

```bash
# Typical usage that solves 80% of use cases
command --flag value
```

For quick problems, this section should provide immediate solution.

## Workflow

Numbered steps for typical usage pattern:

1. **Step one**: Description of what to do
2. **Step two**: Description of what to do
3. **Step three**: Description of what to do
4. **Step four**: Description of what to do

Keep steps focused and actionable.

## Implementation Details

Add section-specific content here. Examples:

### Common Patterns

Explain patterns or approaches:

```bash
# Pattern 1: Basic usage
command input

# Pattern 2: Advanced usage
command --advanced input | filter
```

### Checklist

Use checklists for verification:

- [ ] Item to verify
- [ ] Another item
- [ ] Final item

### Error Handling

Use tables for quick problem identification:

| Problem | Solution |
|---------|----------|
| Specific error message | How to fix it |
| Another error | Another solution |

### Examples

Show practical examples:

```bash
./script 42
./script 73 --verbose
```

## References

Link to comprehensive documentation rather than duplicating:

- See CLAUDE.md for complete [topic] documentation
- See `/agents/guides/topic.md` for detailed guide
- See related skill: gh-another-skill

---

## Format Guidelines

### YAML Frontmatter (Required)

All skills must have:

- `name`: Lowercase, kebab-case identifier
- `description`: One sentence with "Use when" trigger
- `category`: Skill category (github, mojo, documentation, etc.)
- `mcp_fallback`: MCP server fallback (if applicable)

### Section Order

1. Skill name (h1)
2. One-sentence purpose
3. "When to Use" (bullets)
4. "Quick Reference" (code block)
5. "Workflow" (numbered steps)
6. Implementation-specific sections
7. "References" (links to comprehensive docs)

### Line Count Targets

- **Focused skills**: < 100 lines
- **Moderate skills**: < 120 lines
- **Complex skills**: < 150 lines

### Key Principles

1. **Progressive Disclosure**: Most important info first
2. **No Duplication**: Link to comprehensive docs instead
3. **Actionable**: Every step must be doable
4. **Consistent**: Match format of other skills
5. **Concise**: Remove verbose explanations

### When NOT to Duplicate

Do NOT include in skills:

- Comprehensive language reference (link to CLAUDE.md instead)
- Full API documentation (link to official docs)
- Extended tutorial content (link to guides instead)
- Template code (link to template directory)
- Software design patterns (link to architecture docs)

### When TO Use References

Always use reference links for:

- Language-specific syntax rules
- Framework/tool configuration
- Architectural decisions
- Comprehensive guides
- Official documentation

### Code Block Formatting

Bash examples:

```bash
# Always include comments
command --flag value
```

Python examples:

```python
# Python code examples
def function():
    pass
```

Mojo examples:

```mojo
# Mojo code examples
fn function():
    pass
```

Plain text examples:

```text
Plain text output and examples
```

### Error Handling Tables

Format for consistency:

```markdown
| Problem | Solution |
|---------|----------|
| Error message | Specific fix |
| Another error | Another fix |
```

### Cross-Skill References

Reference related skills when helpful:

- See `gh-another-skill` skill for comment retrieval
- See `mojo-test-runner` skill for running tests
- See `doc-generate-adr` skill for creating ADRs

### Verification Checklists

Use for post-task verification:

```markdown
- [ ] All checks passing
- [ ] No new errors
- [ ] Documentation updated
- [ ] Tests passing
```

## Template Variations

### For GitHub Skills

Add MCP fallback:

```yaml
mcp_fallback: github
```

### For Mojo Skills

Reference Mojo language guide:

```markdown
See CLAUDE.md > Mojo Syntax Standards
```

### For Documentation Skills

Link to documentation standards:

```markdown
See CLAUDE.md > Markdown Standards
```

## Common Mistakes to Avoid

1. ❌ Duplicating content from CLAUDE.md → ✅ Link instead
2. ❌ Verbose explanations → ✅ Keep concise with examples
3. ❌ Burying quick reference → ✅ Put after "When to Use"
4. ❌ Inconsistent error format → ✅ Use tables
5. ❌ Missing MCP fallback → ✅ Always specify if applicable
6. ❌ Unclear trigger conditions → ✅ Make "When to Use" specific
7. ❌ Over 150 lines → ✅ Link to extended docs instead

## Testing Your Skill

Before committing:

- [ ] YAML frontmatter valid (name, description, category, mcp_fallback if needed)
- [ ] Skill purpose clear (one sentence)
- [ ] "When to Use" triggers are specific and actionable
- [ ] Quick Reference solves 80% of use cases
- [ ] Workflow steps are numbered and clear
- [ ] No duplication of CLAUDE.md content
- [ ] Error handling uses tables
- [ ] References point to correct docs
- [ ] Line count acceptable (< 150)
- [ ] Code examples are accurate

## File Structure

Skills are located at:

```text
.claude/skills/
├── skill-name/
│   ├── SKILL.md          # Main skill file (this structure)
│   ├── templates/        # If applicable
│   └── reference.md      # If detailed reference needed
```

## Maintaining Consistency

To maintain format consistency across all skills:

1. Copy this template when creating new skills
2. Follow the section order exactly
3. Use same error table format
4. Use same code block formatting
5. Reference comprehensive docs consistently
6. Keep line counts within target ranges

## Quick Checklist

Before submitting new skill:

```bash
# Format check
grep -E "^---" SKILL.md          # Has YAML frontmatter
grep "^# " SKILL.md              # Has h1 heading
grep "## When to Use" SKILL.md   # Has When to Use
grep "## Quick Reference" SKILL.md # Has Quick Reference

# Line count
wc -l SKILL.md                   # Should be < 150

# Format verification
yamllint SKILL.md                # Valid YAML
```

---

This template serves as both a format guide and a working example of the new optimized skill format.
When creating new skills, use this as a reference for structure, length, and content organization.
