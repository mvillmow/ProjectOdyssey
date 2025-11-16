---
name: plan-validate-structure
description: Validate plan directory structure and file format compliance with Template 1 (9-section format). Use before committing plan changes or creating issues.
---

# Plan Structure Validation Skill

Validate plan structure follows Template 1 format.

## When to Use

- After creating new plans
- Before committing plan changes
- Before creating GitHub issues
- Troubleshooting plan errors

## Validation Checks

### 1. Structure

- All 9 sections present
- Sections in correct order
- Proper markdown formatting
- Links use relative paths

### 2. Hierarchy

- Parent/child links valid
- No circular references
- Correct level nesting
- All references exist

### 3. Content

- Title present (# heading)
- Overview not empty
- Steps numbered correctly
- Success criteria have checkboxes

## Usage

```bash
# Validate all plans
./scripts/validate_all_plans.sh

# Validate specific section
./scripts/validate_plans.sh notes/plan/01-foundation

# Validate single plan
./scripts/validate_plan.sh notes/plan/01-foundation/plan.md
```

## Common Issues

### Missing Sections

```text
❌ Missing section: ## Success Criteria
```

**Fix:** Add missing section to plan.md

### Invalid Links

```text
❌ Broken link: [../nonexistent.md](../nonexistent.md)
```

**Fix:** Correct link path or create referenced file

### Wrong Format

```text
❌ Success criteria must use checkboxes (- [ ])
```

**Fix:**
```markdown
## Success Criteria

- [ ] Criterion 1
- [ ] Criterion 2
```

## Template 1 Format

Required sections:
1. Title (# heading)
2. Overview
3. Parent Plan
4. Child Plans
5. Inputs
6. Outputs
7. Steps
8. Success Criteria
9. Notes

See `phase-plan-generate` skill for complete template.
