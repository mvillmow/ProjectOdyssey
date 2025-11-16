---
name: doc-issue-readme
description: Generate issue-specific documentation in /notes/issues/<number>/README.md following ML Odyssey format. Use when starting work on an issue to create proper documentation structure.
---

# Issue README Generation Skill

This skill generates issue-specific documentation following ML Odyssey's standard format.

## When to Use

- User asks to create issue documentation (e.g., "create docs for issue #42")
- Starting work on a GitHub issue
- Need structured documentation for issue
- Tracking implementation progress

## Documentation Location

**All issue-specific outputs go to**: `/notes/issues/<issue-number>/README.md`

## README Format

```markdown
# Issue #XX: [Phase] Component Name

## Objective

What this specific issue accomplishes (1-2 sentences)

## Deliverables

- List of files/changes this issue creates
- Specific, measurable outputs

## Success Criteria

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## References

- Links to /agents/ documentation
- Links to /notes/review/ specifications
- Related issues

## Implementation Notes

Notes discovered during implementation (initially empty)

## Testing Notes

Test results and coverage (if applicable)

## Review Feedback

PR review feedback and resolutions (if applicable)
```

## Usage

### Generate README

```bash
# Generate from issue number
./scripts/generate_issue_readme.sh 42

# This:
# 1. Fetches issue details from GitHub
# 2. Creates /notes/issues/42/ directory
# 3. Generates README.md with template
# 4. Populates with issue information
```

### Manual Creation

```bash
# Create directory
mkdir -p notes/issues/42

# Create README from template
cp .claude/skills/doc-issue-readme/templates/issue_readme.md notes/issues/42/README.md

# Edit to fill in details
```

## README Sections

### 1. Title

Format: `# Issue #XX: [Phase] Component Name`

Examples:
- `# Issue #42: [Implementation] Tensor Operations`
- `# Issue #73: [Test] Neural Network Tests`
- `# Issue #105: [Package] Distribution Archive`

### 2. Objective

**Be specific** - What does this issue accomplish?

Good:
- "Implement basic tensor operations (add, multiply, matmul) with SIMD optimization"
- "Create comprehensive test suite for neural network layers"

Bad:
- "Work on tensors" (too vague)
- "Fix stuff" (not descriptive)

### 3. Deliverables

**List specific files/outputs:**

```markdown
## Deliverables

- `src/tensor/ops.mojo` - Tensor operations implementation
- `tests/test_tensor_ops.mojo` - Comprehensive test suite
- `examples/tensor_demo.mojo` - Usage examples
- `notes/issues/42/performance-results.md` - Benchmark results
```

### 4. Success Criteria

**Measurable checkboxes:**

```markdown
## Success Criteria

- [ ] All tensor operations implemented
- [ ] Tests passing with >90% coverage
- [ ] Performance benchmarks meet requirements
- [ ] Documentation complete
- [ ] PR approved and merged
```

### 5. References

**Link to existing docs, don't duplicate:**

```markdown
## References

- Implementation guide: /agents/implementation-specialist.md
- Testing strategy: /notes/review/testing-strategy.md
- Related issue: #41
- GitHub issue: https://github.com/org/repo/issues/42
```

### 6. Implementation Notes

**Track discoveries during work:**

```markdown
## Implementation Notes

2024-11-15: Started implementation
- Used SIMD width of 8 for float32
- Discovered issue with matrix alignment, added padding

2024-11-16: Completed core functionality
- All basic operations working
- Need to optimize matrix multiplication

2024-11-17: Performance optimization
- Applied loop tiling for cache efficiency
- 3x speedup on large matrices
```

## Documentation Rules

### DO

- ✅ Keep issue-specific
- ✅ Link to comprehensive docs
- ✅ Update as work progresses
- ✅ Be specific and measurable
- ✅ Track actual progress

### DON'T

- ❌ Duplicate comprehensive docs
- ❌ Create shared specifications here
- ❌ Write generic information
- ❌ Leave sections empty long-term
- ❌ Forget to update

## Examples

**Generate for new issue:**
```bash
./scripts/generate_issue_readme.sh 42
```

**Update with progress:**
```bash
# Edit README to add implementation notes
echo "2024-11-15: Completed tensor add operation" >> notes/issues/42/README.md
```

**Check completeness:**
```bash
./scripts/check_issue_docs.sh 42
```

## Scripts Available

- `scripts/generate_issue_readme.sh` - Generate README from issue
- `scripts/check_issue_docs.sh` - Verify README completeness
- `scripts/update_issue_progress.sh` - Add progress notes

## Templates

- `templates/issue_readme.md` - Standard README template
- `templates/implementation_notes.md` - Implementation notes section
- `templates/test_notes.md` - Testing notes section

## Integration with Workflow

### Issue Start

```bash
# 1. Fetch issue
gh issue view 42

# 2. Generate documentation
./scripts/generate_issue_readme.sh 42

# 3. Read and understand
cat notes/issues/42/README.md

# 4. Start implementation
```

### During Work

```bash
# Update notes regularly
vim notes/issues/42/README.md

# Add discoveries, decisions, blockers
```

### Before PR

```bash
# Verify documentation complete
./scripts/check_issue_docs.sh 42

# Ensure all sections filled in
```

## Success Criteria

- [ ] README.md exists in `/notes/issues/<number>/`
- [ ] All sections filled in
- [ ] Deliverables list is specific
- [ ] Success criteria are measurable
- [ ] References link (not duplicate) shared docs
- [ ] Implementation notes track progress

See CLAUDE.md for complete documentation organization rules.
