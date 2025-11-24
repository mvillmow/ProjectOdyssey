# Issue #696: [Plan] Write PR Process - Design and Documentation

## Objective

Document the pull request process including requirements, review process, and what makes a good PR. This helps
contributors submit high-quality PRs that can be reviewed and merged efficiently.

## Deliverables

- PR process section in CONTRIBUTING.md
- PR requirements checklist
- Review process explanation
- Guidelines for good PR descriptions
- Tips for addressing review feedback

## Success Criteria

- [ ] PR process is clearly documented
- [ ] Requirements are specific and actionable
- [ ] Review process is transparent
- [ ] Guidelines help create good PRs
- [ ] Examples illustrate best practices

## Design Decisions

### 1. Documentation Structure

**Decision**: Expand the existing PR section in CONTRIBUTING.md rather than creating a separate document.

### Rationale

- CONTRIBUTING.md already contains a "Pull Request Process" section (lines 191-251)
- Contributors expect to find PR guidance in CONTRIBUTING.md
- Keeping related information together reduces navigation overhead
- Easier to maintain when everything is in one place

### Alternatives Considered

- Create separate PULL_REQUEST.md - Rejected: Adds fragmentation, harder to find
- Create docs/pr-guide.md - Rejected: Duplicates content from CONTRIBUTING.md
- Use GitHub PR template only - Rejected: Not comprehensive enough for process documentation

### 2. Content Organization

**Decision**: Structure PR section into clear subsections covering the complete lifecycle:

1. Before You Start (preparation)
1. Creating Your Pull Request (submission)
1. Code Review (addressing feedback)
1. Merging (completion)

### Rationale

- Mirrors the natural workflow of creating a PR
- Easy to find information for each stage
- Reduces cognitive load by separating concerns
- Supports both new and experienced contributors

### Alternatives Considered

- Single narrative flow - Rejected: Hard to scan for specific information
- FAQ-style Q&A - Rejected: Doesn't match natural workflow
- Checklist-only format - Rejected: Missing context and explanations

### 3. Review Comment Handling

**Decision**: Include specific commands for replying to review comments using GitHub API.

### Rationale

- CLAUDE.md already documents this pattern (lines 134-165)
- Common source of confusion (general comments vs review replies)
- Explicit commands prevent mistakes
- Matches existing project conventions

**Current Coverage**: The existing CONTRIBUTING.md already includes this (lines 223-241), but could be enhanced with:

- More examples of good reply messages
- Explanation of why individual replies matter
- Visual distinction between PR comments and review replies

### 4. PR Quality Guidelines

**Decision**: Add section on what makes a good PR description.

### Rationale

- Currently missing from CONTRIBUTING.md
- Directly impacts review efficiency
- Reduces back-and-forth communication
- Helps PRs get merged faster

### Content to Include

- Clear title format (conventional commits style already used)
- Description structure (context, changes, testing)
- When to split PRs (scope management)
- How to reference issues and related PRs
- Examples of good vs poor descriptions

### 5. PR Requirements Checklist

**Decision**: Create an actionable checklist that covers technical and process requirements.

### Rationale

- Makes requirements explicit and verifiable
- Reduces reviewer burden
- Helps contributors self-check before submission
- Can be used in PR templates

### Checklist Sections

- Code quality (tests, formatting, linting)
- Documentation (code comments, README updates)
- Process (issue linked, branch naming, commits)
- CI/CD (all checks passing)

### 6. Review Process Transparency

**Decision**: Document what reviewers look for and how long reviews typically take.

### Rationale

- Reduces uncertainty for contributors
- Sets appropriate expectations
- Helps contributors prepare better PRs
- Makes the process feel fair and predictable

### Content to Include

- Review criteria (correctness, clarity, maintainability, tests)
- Typical review timeline
- What happens after approval
- How to escalate if review is stalled

### 7. Handling Review Feedback

**Decision**: Provide specific guidance on addressing different types of feedback.

### Rationale

- Common point of confusion for contributors
- Different feedback types require different responses
- Explicit guidance reduces mistakes

### Feedback Types

- Required changes (blocking)
- Suggestions (non-blocking)
- Questions (need clarification)
- Nitpicks (style/preference)

### 8. Integration with Existing Documentation

**Decision**: Link to CLAUDE.md for detailed technical workflows, keep CONTRIBUTING.md focused on contributor-facing guidance.

### Rationale

- Avoid duplication between CONTRIBUTING.md and CLAUDE.md
- CONTRIBUTING.md is for all contributors (external + internal)
- CLAUDE.md is for Claude Code automation and internal workflows
- Cross-reference where needed

**Current State**: CONTRIBUTING.md already references CLAUDE.md for architecture (line 335)

## References

- Source Plan: `/notes/plan/01-foundation/03-initial-documentation/02-contributing/03-write-pr-process/plan.md`
- Parent Plan: `/notes/plan/01-foundation/03-initial-documentation/02-contributing/plan.md`
- Current CONTRIBUTING.md: Lines 191-251 (existing PR section)
- CLAUDE.md Git Workflow: Lines 134-165 (review comment handling)
- Related Issues:
  - #697 - [Test] Write PR Process
  - #698 - [Impl] Write PR Process
  - #699 - [Package] Write PR Process
  - #700 - [Cleanup] Write PR Process

## Implementation Notes

This section will be filled during the implementation phase (#698).

### Key Considerations for Implementation

1. **Preserve Existing Content**: The PR section already has valuable information (lines 191-251)
1. **Add Missing Pieces**: Focus on what's not covered yet (PR quality guidelines, review criteria)
1. **Enhance What Exists**: Improve review comment handling with more examples
1. **Maintain Consistency**: Match tone and style of existing CONTRIBUTING.md
1. **Visual Clarity**: Use proper markdown formatting (code blocks, lists, headings)

### Sections to Add/Enhance

- **New**: "Writing a Good PR Description" section
- **New**: "PR Requirements Checklist" section
- **Enhance**: Review comment examples with more context
- **New**: "What Reviewers Look For" section
- **Enhance**: Add timing expectations to review process
- **New**: "Addressing Different Types of Feedback" section

### Testing Considerations

Testing phase (#697) should verify:

- All links work correctly
- Code examples are accurate and run successfully
- Markdown linting passes
- Documentation is accessible to target audience
- Examples illustrate best practices effectively
