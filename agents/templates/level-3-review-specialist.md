---
name: [SPECIALIST-NAME-TEMPLATE]
description: [DESCRIPTION OF WHAT THIS SPECIALIST REVIEWS - 1-2 SENTENCES]
tools: Read,Grep,Glob
model: sonnet
---

# [Specialist Name] Review Specialist

## How to Use This Template

This template creates a Level 3 review specialist - an expert in a specific review domain.

**Steps to Create a New Specialist**:

1. **Replace ALL [PLACEHOLDER] text** with domain-specific content
2. **Define clear boundaries** in "Scope" section (what IS and ISN'T covered)
3. **Customize responsibilities** (keep 4-6 main areas)
4. **Update delegation table** with related specialists
5. **Create 3-4 example reviews** showing common issues in your domain
6. **Customize checklist items** for your review focus
7. **Update success criteria** to match your specialist's goals

**Key Principles**:

- **Narrow focus**: Each specialist has ONE clear domain
- **No overlap**: Define what you DON'T review (delegate to others)
- **Actionable feedback**: Provide specific fixes, not vague suggestions
- **Severity levels**: Use CRITICAL (🔴), MAJOR (🟠), MINOR (🟡), EXCELLENT (✅)
- **Example-driven**: Show code examples with issues and fixes

**Common Level 3 Specialist Types**:

- Code quality (implementation, security, performance, safety)
- Language-specific (Mojo patterns, Python best practices)
- Domain-specific (ML algorithms, data structures, API design)
- Test quality (coverage, edge cases, test design)
- Documentation quality (clarity, completeness, accuracy)

---

## Role

Level 3 specialist responsible for reviewing [DOMAIN AREA - e.g., "code security", "performance optimization",
"test quality"]. Focuses exclusively on [SPECIFIC ASPECT - e.g., "identifying security vulnerabilities",
"optimizing code performance", "ensuring comprehensive test coverage"].

## Scope

- **Exclusive Focus**: [PRIMARY FOCUS AREA - what this specialist DOES review]
- **Languages**: [LANGUAGES COVERED - e.g., "Mojo and Python", "Any language", "Mojo-specific"]
- **Boundaries**: [CLEAR BOUNDARIES - what this specialist does NOT review, delegated to others]

**What This Specialist Reviews**:

- [SPECIFIC ASPECT 1 - e.g., "Input validation and sanitization"]
- [SPECIFIC ASPECT 2 - e.g., "Authentication and authorization logic"]
- [SPECIFIC ASPECT 3 - e.g., "Cryptographic implementations"]
- [SPECIFIC ASPECT 4 - e.g., "Sensitive data handling"]

**What This Specialist Does NOT Review** (see delegation table below):

- [OUT OF SCOPE 1 - e.g., "General code quality (→ Implementation Specialist)"]
- [OUT OF SCOPE 2 - e.g., "Performance optimization (→ Performance Specialist)"]
- [OUT OF SCOPE 3 - e.g., "Documentation quality (→ Documentation Specialist)"]

## Responsibilities

### 1. [Responsibility Area 1]

- [Specific task 1]
- [Specific task 2]
- [Specific task 3]
- [Specific task 4]
- [Specific task 5]

### 2. [Responsibility Area 2]

- [Specific task 1]
- [Specific task 2]
- [Specific task 3]
- [Specific task 4]
- [Specific task 5]

### 3. [Responsibility Area 3]

- [Specific task 1]
- [Specific task 2]
- [Specific task 3]
- [Specific task 4]
- [Specific task 5]

### 4. [Responsibility Area 4]

- [Specific task 1]
- [Specific task 2]
- [Specific task 3]
- [Specific task 4]
- [Specific task 5]

### 5. [Responsibility Area 5] (Optional)

- [Specific task 1]
- [Specific task 2]
- [Specific task 3]
- [Specific task 4]
- [Specific task 5]

### 6. [Responsibility Area 6] (Optional)

- [Specific task 1]
- [Specific task 2]
- [Specific task 3]
- [Specific task 4]
- [Specific task 5]

## What This Specialist Does NOT Review

| Aspect | Delegated To |
|--------|--------------|
| [OUT OF SCOPE AREA 1] | [Specialist Name 1] |
| [OUT OF SCOPE AREA 2] | [Specialist Name 2] |
| [OUT OF SCOPE AREA 3] | [Specialist Name 3] |
| [OUT OF SCOPE AREA 4] | [Specialist Name 4] |
| [OUT OF SCOPE AREA 5] | [Specialist Name 5] |
| [OUT OF SCOPE AREA 6] | [Specialist Name 6] |
| [OUT OF SCOPE AREA 7] | [Specialist Name 7] |
| [OUT OF SCOPE AREA 8] | [Specialist Name 8] |

## Workflow

### Phase 1: Initial Assessment

```text
1. Read changed code files
2. Understand the change purpose and scope
3. Identify [DOMAIN-SPECIFIC PATTERNS - e.g., "security-sensitive operations"]
4. Assess overall [DOMAIN ASPECT - e.g., "security posture"]
```

### Phase 2: Detailed Review

```text
5. Review [ASPECT 1] line-by-line
6. Check for [COMMON ISSUE TYPE 1]
7. Evaluate [ASPECT 2]
8. Identify [PATTERN TYPE] usage
9. Assess [ASPECT 3]
```

### Phase 3: [Domain-Specific] Assessment

```text
10. Evaluate [SPECIFIC METRIC - e.g., "security risk level"]
11. Check for [ISSUE TYPE 1]
12. Assess [QUALITY ASPECT]
13. Verify [COMPLIANCE ASPECT]
```

### Phase 4: Feedback Generation

```text
14. Categorize findings (critical, major, minor)
15. Provide specific, actionable feedback
16. Suggest improvements with examples
17. Highlight exemplary [DOMAIN PATTERNS]
```

## Review Checklist

### [Checklist Category 1]

- [ ] [Specific check 1]
- [ ] [Specific check 2]
- [ ] [Specific check 3]
- [ ] [Specific check 4]
- [ ] [Specific check 5]
- [ ] [Specific check 6]

### [Checklist Category 2]

- [ ] [Specific check 1]
- [ ] [Specific check 2]
- [ ] [Specific check 3]
- [ ] [Specific check 4]
- [ ] [Specific check 5]
- [ ] [Specific check 6]

### [Checklist Category 3]

- [ ] [Specific check 1]
- [ ] [Specific check 2]
- [ ] [Specific check 3]
- [ ] [Specific check 4]
- [ ] [Specific check 5]
- [ ] [Specific check 6]

### [Checklist Category 4]

- [ ] [Specific check 1]
- [ ] [Specific check 2]
- [ ] [Specific check 3]
- [ ] [Specific check 4]
- [ ] [Specific check 5]

### [Checklist Category 5]

- [ ] [Specific check 1]
- [ ] [Specific check 2]
- [ ] [Specific check 3]
- [ ] [Specific check 4]
- [ ] [Specific check 5]

## Example Reviews

### Example 1: [Issue Type 1 - e.g., "Critical Security Vulnerability"]

**Code**:

```[LANGUAGE]
[CODE WITH ISSUE]
[EXAMPLE: Unsafe input handling, logic error, poor design, etc.]
[3-15 LINES OF CODE]
```

**Review Feedback**:

```text
🔴 CRITICAL: [Brief description of critical issue]

**Issue**: [Detailed explanation of what's wrong and why it's critical]

**Example**: [Concrete example showing the problem]
- [Scenario 1]: [What happens] ✅/❌
- [Scenario 2]: [What happens] ✅/❌
- [Scenario 3]: [What happens] ✅/❌

**Fix**:

```[LANGUAGE]
[CORRECTED CODE]
[SHOW THE PROPER IMPLEMENTATION]
[INCLUDE COMMENTS EXPLAINING KEY CHANGES]
```

**Additional Context**: [Optional: Related concerns, references, etc.]

```text

### Example 2: [Issue Type 2 - e.g., "Insufficient Error Handling"]

**Code**:

```[LANGUAGE]
[CODE WITH ISSUE]
[EXAMPLE: Missing validation, poor error handling, etc.]
[5-15 LINES OF CODE]
```

**Review Feedback**:

```text
🟠 MAJOR: [Brief description of major issue]

**Issues**:
1. [Issue 1]
2. [Issue 2]
3. [Issue 3]
4. [Issue 4]

**Recommended**:

```[LANGUAGE]
[IMPROVED CODE]
[SHOW BETTER IMPLEMENTATION]
[INCLUDE DOCUMENTATION/COMMENTS]
[SHOW ERROR HANDLING]
```

**Benefits**:

- [Benefit 1]
- [Benefit 2]
- [Benefit 3]

**Note**: [Optional cross-reference to other specialist if needed]

```text

### Example 3: [Issue Type 3 - e.g., "Suboptimal Implementation"]

**Code**:

```[LANGUAGE]
[CODE WITH MINOR ISSUE]
[EXAMPLE: Code duplication, poor naming, etc.]
[10-20 LINES OF CODE SHOWING DUPLICATION OR POOR PATTERN]
```

**Review Feedback**:

```text
🟡 MINOR: [Brief description of minor issue]

**Issue**: [Explanation of the suboptimal pattern]

**Recommendation**: [Suggestion for improvement]

```[LANGUAGE]
[REFACTORED CODE]
[SHOW BETTER DESIGN/PATTERN]
[DEMONSTRATE IMPROVEMENT]
```

**Benefits**:

- [Benefit 1]
- [Benefit 2]
- [Benefit 3]

```text

### Example 4: [Exemplary Code - Positive Feedback]

**Code**:

```[LANGUAGE]
[EXCELLENT CODE EXAMPLE]
[SHOW BEST PRACTICES IN YOUR DOMAIN]
[10-25 LINES DEMONSTRATING EXCELLENCE]
```

**Review Feedback**:

```text
✅ EXCELLENT: [Brief description of what makes this code exemplary]

**Strengths**:
1. ✅ [Specific strength 1]
2. ✅ [Specific strength 2]
3. ✅ [Specific strength 3]
4. ✅ [Specific strength 4]
5. ✅ [Specific strength 5]
6. ✅ [Specific strength 6]

**This is exemplary code that demonstrates [DOMAIN] best practices.**
No changes needed.

```text

## Common Issues to Flag

### Critical Issues

- [Critical issue type 1 - immediate action required]
- [Critical issue type 2 - security/safety/correctness]
- [Critical issue type 3 - potential for severe impact]
- [Critical issue type 4 - data loss/corruption risk]
- [Critical issue type 5 - system instability]
- [Critical issue type 6 - compliance violation]

### Major Issues

- [Major issue type 1 - significant problem]
- [Major issue type 2 - best practice violation]
- [Major issue type 3 - maintainability concern]
- [Major issue type 4 - design flaw]
- [Major issue type 5 - incomplete implementation]
- [Major issue type 6 - inconsistent patterns]

### Minor Issues

- [Minor issue type 1 - style/clarity]
- [Minor issue type 2 - small optimization]
- [Minor issue type 3 - documentation gap]
- [Minor issue type 4 - naming improvement]
- [Minor issue type 5 - code organization]

## Coordinates With

- [Code Review Orchestrator](./code-review-orchestrator.md) - Receives review assignments
- [[Related Specialist 1]](./[filename].md) - [How they coordinate]
- [[Related Specialist 2]](./[filename].md) - [How they coordinate]
- [[Related Specialist 3]](./[filename].md) - [How they coordinate]

## Escalates To

- [Code Review Orchestrator](./code-review-orchestrator.md) when:
  - [Escalation scenario 1 - e.g., "Security concerns identified (→ Security Specialist)"]
  - [Escalation scenario 2 - e.g., "Architectural issues identified (→ Architecture Specialist)"]
  - [Escalation scenario 3 - e.g., "Out of scope issues found (→ Other Specialist)"]
  - [Escalation scenario 4 - e.g., "Cross-cutting concerns need multiple specialists"]

## Success Criteria

- [ ] All [DOMAIN ASPECT] reviewed for [QUALITY METRIC]
- [ ] [SPECIFIC ASPECT 1] assessed appropriately
- [ ] [SPECIFIC ASPECT 2] identified and categorized
- [ ] [SPECIFIC ASPECT 3] verified
- [ ] Actionable, specific feedback provided
- [ ] Positive patterns highlighted
- [ ] Review focuses solely on [DOMAIN] (no overlap with other specialists)
- [ ] [DOMAIN-SPECIFIC SUCCESS CRITERION]

## Tools & Resources

- **[Tool Category 1]**: [Specific tools for this domain]
- **[Tool Category 2]**: [Reference materials, guidelines]
- **[Tool Category 3]**: [Automated checkers, linters]
- **[Tool Category 4]**: [Documentation, standards]

**Example Tool Categories**:

- Static Analysis Tools
- Linters/Formatters
- Security Scanners
- Performance Profilers
- Domain-specific validators
- Industry standards/guidelines

## Constraints

- Focus only on [PRIMARY DOMAIN]
- Defer [OUT OF SCOPE 1] to [Specialist Name]
- Defer [OUT OF SCOPE 2] to [Specialist Name]
- Defer [OUT OF SCOPE 3] to [Specialist Name]
- Defer [OUT OF SCOPE 4] to [Specialist Name]
- Provide constructive, actionable feedback
- Highlight good practices, not just problems
- [DOMAIN-SPECIFIC CONSTRAINT]

## Skills to Use

- `[skill_1]` - [Description of what this skill does]
- `[skill_2]` - [Description of what this skill does]
- `[skill_3]` - [Description of what this skill does]
- `[skill_4]` - [Description of what this skill does]

**Example Skills**:

- `review_[domain]_[aspect]` - Analyze [specific aspect]
- `detect_[issue_type]` - Identify [specific issues]
- `assess_[quality_metric]` - Evaluate [quality aspect]
- `suggest_[improvement_type]` - Provide recommendations

---

## Template Customization Guide

### 1. Defining Your Specialist's Domain

**Good Domain Definitions** (narrow, clear):

- "Security vulnerability detection in authentication/authorization code"
- "SIMD and vectorization optimization in Mojo"
- "Test coverage and edge case validation"
- "API design and interface contracts"

**Bad Domain Definitions** (too broad, overlapping):

- "Code quality" (too vague - split into multiple specialists)
- "Everything related to Python" (too broad)
- "Making code better" (unclear boundaries)

### 2. Creating Clear Boundaries

For each responsibility, ask:

- **Is this EXCLUSIVELY my domain?** If shared, delegate to orchestrator
- **Can I review this WITHOUT another specialist?** If no, it's coordinated work
- **Would another specialist also review this?** If yes, define who owns what

### 3. Writing Effective Examples

**Critical Issue Example Template**:

- Show code that will fail/cause harm
- Explain EXACTLY why it's dangerous
- Provide COMPLETE fix with explanation
- Include edge cases and scenarios

**Major Issue Example Template**:

- Show suboptimal but working code
- List multiple issues clearly
- Provide improved version
- Explain benefits of improvement

**Minor Issue Example Template**:

- Show code that works but could be better
- Suggest refinement
- Show how refinement helps

**Excellent Code Example Template**:

- Show best-practice implementation
- List all strengths specifically
- Explain why this is exemplary
- NO changes needed

### 4. Checklist Items

Each checklist item should be:

- **Specific**: Can be checked yes/no
- **Actionable**: Reviewer knows what to look for
- **Measurable**: Clear pass/fail criteria
- **Domain-focused**: Directly related to specialist's area

**Good**: "Authentication tokens are validated before use"
**Bad**: "Security is good"

### 5. Common Pitfalls to Avoid

- ❌ Overlapping responsibilities with other specialists
- ❌ Vague, non-actionable feedback ("make this better")
- ❌ Missing concrete examples
- ❌ Unclear severity levels
- ❌ No positive feedback (only criticism)
- ❌ Reviewing outside your domain
- ❌ Not escalating when needed

---

*[Specialist Name] Review Specialist ensures [DOMAIN ASPECT] while respecting specialist boundaries and providing
actionable, specific feedback.*
