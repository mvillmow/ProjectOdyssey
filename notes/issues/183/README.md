# Issue #183: [Plan] Write PR Process - Design and Documentation

## Objective

Design comprehensive pull request process documentation for CONTRIBUTING.md that helps contributors submit
high-quality PRs and understand the review process. This planning phase produces detailed specifications that will
guide the implementation (#185), testing (#184), packaging (#186), and cleanup (#187) phases.

## Deliverables

### Primary Deliverable

- **Comprehensive PR Process Design Specification** - Detailed plan covering all aspects of the PR workflow

### Design Components

1. **PR Requirements Specification**
   - Code quality requirements (tests, documentation, style)
   - Pre-submission checklist
   - Required CI checks
   - Documentation requirements

2. **Review Process Documentation**
   - What reviewers look for
   - Review timeline and expectations
   - How to interpret review feedback
   - Escalation process for disagreements

3. **PR Description Guidelines**
   - Template structure
   - Required sections (summary, testing, breaking changes)
   - Examples of good and bad PR descriptions
   - Linking to issues

4. **Review Feedback Handling**
   - How to respond to review comments (inline replies vs general comments)
   - Using GitHub review comment API correctly
   - Verification steps after addressing feedback
   - Re-requesting review after changes

5. **Examples and Best Practices**
   - Example PR descriptions
   - Example review responses
   - Common mistakes to avoid
   - Tips for efficient review cycles

## Success Criteria

- [ ] PR process specification is comprehensive and addresses all workflow stages
- [ ] Requirements are specific, actionable, and verifiable
- [ ] Review process is transparent and sets clear expectations
- [ ] Guidelines provide concrete examples of good practices
- [ ] Examples illustrate both correct and incorrect approaches
- [ ] Documentation supports both new and experienced contributors
- [ ] Specification enables implementation phase (#185) to proceed without ambiguity
- [ ] Testing criteria (#184) can be derived directly from specification

## References

### Existing Documentation

- [CONTRIBUTING.md](/home/mvillmow/ml-odyssey-manual/CONTRIBUTING.md) - Current contribution guide (lines 191-251 cover PR process)
- [GitHub Review Comments Guide](/home/mvillmow/ml-odyssey-manual/agents/guides/github-review-comments.md) - Detailed guide on handling review comments
- [Verification Checklist](/home/mvillmow/ml-odyssey-manual/agents/guides/verification-checklist.md) - Standard verification workflows
- [CLAUDE.md](/home/mvillmow/ml-odyssey-manual/CLAUDE.md) - PR creation instructions (lines 417-467)

### Source Plan

- [Write PR Process Plan](/home/mvillmow/ml-odyssey-manual/notes/plan/01-foundation/03-initial-documentation/02-contributing/03-write-pr-process/plan.md)

### Related Issues

- Issue #184: [Test] Write PR Process - Write Tests
- Issue #185: [Impl] Write PR Process - Implementation
- Issue #186: [Package] Write PR Process - Integration and Packaging
- Issue #187: [Cleanup] Write PR Process - Refactor and Finalize

### Parent Context

- [Contributing Plan](/home/mvillmow/ml-odyssey-manual/notes/plan/01-foundation/03-initial-documentation/02-contributing/plan.md)

## Implementation Notes

### Current State Analysis

The existing CONTRIBUTING.md (lines 191-251) contains a basic PR process section covering:

1. **Before You Start** - Issue linking, branch naming, TDD workflow
2. **Creating Your Pull Request** - Using `gh pr create`, PR template requirements
3. **Code Review** - Review comment handling using GitHub API
4. **Merging** - Post-approval workflow

**Strengths:**

- Covers GitHub CLI usage for PR creation
- Includes correct GitHub API commands for review comment replies
- Links PRs to issues automatically
- Emphasizes verification (checking CI, confirming replies posted)

**Gaps to Address:**

1. **PR Description Guidelines** - No template or examples of good PR descriptions
2. **Review Expectations** - Doesn't explain what reviewers look for or timeline
3. **Common Mistakes** - Could benefit from expanded examples
4. **Pre-submission Checklist** - Missing comprehensive checklist
5. **Breaking Changes** - No guidance on documenting breaking changes
6. **Review Feedback Types** - Could clarify when to use different response types

### Design Decisions

#### 1. PR Description Template

**Decision:** Create a structured template with required sections

**Rationale:** Standardized PR descriptions improve review efficiency and ensure critical information isn't missed.

**Proposed Template:**

```markdown
## Summary
<!-- Brief description of what changed and why -->

## Changes
<!-- Bulleted list of key changes -->

## Testing
<!-- How the changes were tested -->

## Breaking Changes
<!-- Any backwards-incompatible changes, or "None" -->

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Pre-commit hooks pass locally
- [ ] All CI checks pass
```

#### 2. Review Comment Handling

**Decision:** Reference existing comprehensive guide rather than duplicate

**Approach:**

- Keep essential commands in CONTRIBUTING.md
- Link to detailed guide in `/agents/guides/github-review-comments.md`
- Emphasize verification steps

**Rationale:** Avoids documentation duplication while ensuring critical information is easily accessible.

#### 3. Review Process Transparency

**Decision:** Document reviewer expectations and typical timeline

**Content to Add:**

- What reviewers check (correctness, tests, style, documentation)
- Expected response time (e.g., "reviews typically within 48 hours")
- How to escalate if review is delayed
- When re-review is needed

**Rationale:** Setting expectations reduces contributor frustration and improves collaboration.

#### 4. Examples Strategy

**Decision:** Provide both positive and negative examples

**Approach:**

- Good PR description example
- Bad PR description example with explanation
- Good review comment response
- Bad review comment response with explanation

**Rationale:** Showing what NOT to do is often more instructive than just showing best practices.

### Content Structure Plan

#### Section 1: Before Creating a PR

**Content:**

- Check for existing issue (create if needed)
- Branch naming: `<issue-number>-<description>`
- TDD workflow reminder
- Pre-submission checklist

**Location:** Keep in existing "Before You Start" section with enhancements

#### Section 2: PR Description Guidelines

**Content:**

- Template structure
- Required sections explained
- Example of good PR description
- Example of bad PR description with improvements

**Location:** New section after "Creating Your Pull Request"

#### Section 3: Review Process

**Content:**

- What reviewers look for
- Typical timeline and expectations
- How to interpret different types of feedback
- When changes are required vs optional

**Location:** Expand existing "Code Review" section

#### Section 4: Addressing Review Feedback

**Content:**

- Two types of comments (PR-level vs inline review comments)
- Correct API usage for inline replies
- Verification steps (MUST verify replies posted)
- Re-requesting review after changes
- Link to comprehensive guide

**Location:** Enhance existing review section with better structure

#### Section 5: CI and Verification

**Content:**

- Waiting for CI to start (sleep 30)
- Checking CI status
- Local vs CI pre-commit differences
- What to do if CI fails

**Location:** New section or expand in review section

#### Section 6: Merging

**Content:**

- Post-approval workflow
- Branch cleanup
- Squashing commits when appropriate

**Location:** Keep existing content with minor enhancements

### Integration with Existing Documentation

#### Links to Maintain

1. **Detailed Guides:**
   - `/agents/guides/github-review-comments.md` - Comprehensive review comment handling
   - `/agents/guides/verification-checklist.md` - Standard verification workflows

2. **Project Documentation:**
   - `CLAUDE.md` - Agent-specific PR creation instructions
   - `README.md` - Project overview

3. **External Resources:**
   - GitHub PR documentation
   - Conventional commits specification

#### Avoid Duplication

**Do NOT duplicate:**

- Complete command reference (link to guides instead)
- Agent-specific workflows (keep in CLAUDE.md)
- Detailed troubleshooting (link to guides)

**Do include:**

- Essential commands for common workflows
- Quick reference for standard operations
- Links to comprehensive documentation

### Documentation Standards

All content must follow markdown linting rules (CONTRIBUTING.md lines 126-161):

1. **Code Blocks:**
   - Must specify language (` ```bash `, not ` ``` `)
   - Must have blank lines before and after

2. **Lists:**
   - Must have blank lines before and after

3. **Headings:**
   - Must have blank lines before and after

4. **Line Length:**
   - Maximum 120 characters
   - Break at natural boundaries

5. **Links:**
   - Use relative paths when possible
   - Use reference-style for long URLs

### Testing Approach (for Issue #184)

The test phase should verify:

1. **Completeness:**
   - All required sections present
   - Examples provided for each guideline
   - Links to related documentation work

2. **Accuracy:**
   - Commands are correct and tested
   - API endpoints match GitHub documentation
   - Examples reflect current repository practices

3. **Clarity:**
   - Instructions are unambiguous
   - Examples are relevant
   - Language is accessible to newcomers

4. **Consistency:**
   - Terminology consistent with other docs
   - Format matches project standards
   - Tone is welcoming and professional

**Test Method:** Manual review against checklist (documentation testing)

### Implementation Approach (for Issue #185)

The implementation phase should:

1. **Preserve Existing Content:**
   - Keep working sections (branch naming, GitHub CLI usage)
   - Enhance rather than replace

2. **Add New Sections:**
   - PR description guidelines with template
   - Review expectations and timeline
   - Expanded verification steps

3. **Improve Examples:**
   - Add good/bad PR description examples
   - Add good/bad review response examples
   - Include real scenarios from project history

4. **Structure for Readability:**
   - Use clear headings
   - Break long sections into subsections
   - Use formatting (bold, lists) effectively

### Packaging Approach (for Issue #186)

The packaging phase should verify:

1. **Integration:**
   - Links to other documentation work correctly
   - Cross-references are accurate
   - No broken links

2. **Navigation:**
   - Table of contents (if needed)
   - Clear section hierarchy
   - Easy to find specific information

3. **Accessibility:**
   - Works for both new and experienced contributors
   - Progressive disclosure (basics first, details later)

### Cleanup Approach (for Issue #187)

The cleanup phase should:

1. **Review for Redundancy:**
   - Remove duplicated information
   - Consolidate similar sections
   - Ensure DRY principle

2. **Polish Language:**
   - Consistent tone
   - Clear phrasing
   - No jargon without explanation

3. **Final Verification:**
   - All links work
   - All examples tested
   - Markdown linting passes

## Design Specification

### 1. PR Requirements Specification

#### 1.1 Code Quality Requirements

**Mandatory Requirements:**

1. **Tests:**
   - All new functionality must have tests
   - Existing tests must pass
   - Test coverage should not decrease (aim for >80% on new code)
   - Follow TDD: write tests before implementation

2. **Documentation:**
   - Public APIs must have docstrings
   - README updates for new features
   - CHANGELOG entry for notable changes
   - Comments for complex logic

3. **Code Style:**
   - Mojo: Use `mojo format` (enforced by pre-commit)
   - Python: Follow PEP 8 with `black` formatting
   - Markdown: Pass `markdownlint-cli2`

4. **Pre-commit Hooks:**
   - All hooks must pass locally
   - CI pre-commit check must pass
   - Do NOT use `--no-verify` unless documented why

#### 1.2 Pre-submission Checklist

**Before Creating PR:**

- [ ] Issue exists and is linked
- [ ] Branch follows naming: `<issue-number>-<description>`
- [ ] Tests written and passing locally
- [ ] Documentation updated
- [ ] Pre-commit hooks pass: `pre-commit run --all-files`
- [ ] Code follows project standards (Mojo: `fn` over `def`, etc.)
- [ ] Commits follow conventional commit format

**In PR Description:**

- [ ] Summary explains what and why
- [ ] Changes are listed
- [ ] Testing approach described
- [ ] Breaking changes documented (or "None")
- [ ] Issue linked using `gh pr create --issue <number>` or "Closes #123"

#### 1.3 CI Check Requirements

**All PRs Must Pass:**

1. **Pre-commit** - Code formatting and linting
2. **Tests** - All test suites pass
3. **Build** - Code compiles/builds successfully

**Before Merging:**

- All CI checks must be green
- At least one approval from maintainer
- No unresolved review comments

### 2. Review Process Documentation

#### 2.1 What Reviewers Look For

**Correctness:**

- Does the code solve the problem correctly?
- Are edge cases handled?
- Are there potential bugs?

**Testing:**

- Are there tests for new functionality?
- Do tests cover edge cases?
- Are tests clear and maintainable?

**Code Quality:**

- Is the code readable and maintainable?
- Are names descriptive?
- Is complexity justified?
- Is there unnecessary duplication?

**Documentation:**

- Are public APIs documented?
- Are complex sections explained?
- Is README updated if needed?

**Project Standards:**

- Follows Mojo best practices (for Mojo code)
- Follows Python standards (for Python code)
- Consistent with existing codebase
- Minimal changes principle followed

#### 2.2 Review Timeline and Expectations

**Typical Timeline:**

- **Initial review:** Within 48 hours of PR creation
- **Follow-up review:** Within 24 hours of addressing comments
- **Merge:** Same day as final approval if CI passes

**What to Expect:**

- Reviewers may request changes
- Multiple rounds of review are normal
- Some feedback is required, some is optional (marked "nit:")
- Questions are welcome - ask for clarification

**If Review is Delayed:**

- Tag the reviewer after 48 hours
- Use GitHub notifications
- Mention in project chat/discussion (if available)

#### 2.3 Interpreting Review Feedback

**Types of Feedback:**

1. **Required Changes:**
   - Must be addressed before merge
   - Usually about correctness, tests, or critical issues
   - Blocking approval

2. **Optional Suggestions (marked "nit:"):**
   - Nice to have but not required
   - Style preferences
   - Non-blocking

3. **Questions:**
   - Reviewer seeking clarification
   - May lead to required or optional changes
   - Respond even if no code change needed

**Reading Between the Lines:**

- "Consider..." = optional suggestion
- "Should..." or "Must..." = required change
- "Why...?" = question, respond with explanation
- "nit:" = minor/optional

#### 2.4 Escalation Process

**When to Escalate:**

- Disagreement on technical approach
- Review delayed beyond reasonable time
- Unclear requirements
- Conflicting feedback from multiple reviewers

**How to Escalate:**

1. First: Ask for clarification in PR comments
2. Second: Tag project maintainer
3. Third: Create discussion issue for technical decisions

### 3. PR Description Guidelines

#### 3.1 Template Structure

```markdown
## Summary
[2-3 sentence description of what changed and why]

## Changes
- Change 1
- Change 2
- Change 3

## Testing
[How you tested: manual tests, automated tests, etc.]

## Breaking Changes
[Any backwards-incompatible changes, or "None"]

## Additional Context
[Optional: links, background info, related issues]

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Pre-commit hooks pass locally
- [ ] All CI checks pass
```

#### 3.2 Section Descriptions

**Summary:**

- **Purpose:** Quick overview for reviewers
- **Content:** What changed, why it changed
- **Length:** 2-3 sentences
- **Example:** "Implemented environment variable substitution in config loader to support dynamic configuration. Uses `${VAR_NAME}` syntax with `os.path.expandvars()` for compatibility with existing Python patterns."

**Changes:**

- **Purpose:** Bulleted list of key modifications
- **Content:** Files changed, new features, bug fixes
- **Format:** Bullet points, grouped by category if needed
- **Example:**
  - Added `expand_env_vars()` helper function
  - Updated config parser to call expansion on all string values
  - Added tests for environment variable substitution

**Testing:**

- **Purpose:** Prove the changes work
- **Content:** Test types, manual testing done, edge cases covered
- **Example:** "Added 5 unit tests covering environment variable expansion, missing variables, and nested expansion. Manually tested with actual config files."

**Breaking Changes:**

- **Purpose:** Alert users to compatibility issues
- **Content:** What breaks, migration path, or "None"
- **Example:** "None" OR "Changed config loading to require explicit env var expansion. Old configs work unchanged."

**Checklist:**

- **Purpose:** Self-verification before review
- **Content:** Standard checks from pre-submission checklist
- **All items must be checked before requesting review**

#### 3.3 Examples

**Example 1: Good PR Description**

```markdown
## Summary
Implemented environment variable substitution in the configuration loader to support dynamic configuration values.
This allows configs to reference environment variables using `${VAR_NAME}` syntax, enabling environment-specific
settings without duplicating config files.

## Changes
- Added `expand_env_vars()` helper function in `utils/config.py`
- Updated YAML/JSON parsers to expand environment variables in string values
- Added comprehensive tests for variable expansion edge cases
- Updated documentation with usage examples

## Testing
- Added 5 unit tests covering:
  - Basic environment variable expansion
  - Missing variables (raises ConfigError)
  - Nested expansion (not supported, documented)
  - Empty variable values
  - Special characters in values
- Manually tested with sample config files
- All existing tests pass

## Breaking Changes
None - existing configs work unchanged. Variable expansion is opt-in via `${VAR}` syntax.

## Checklist
- [x] Tests added/updated
- [x] Documentation updated
- [x] Pre-commit hooks pass locally
- [x] All CI checks pass
```

**Example 2: Bad PR Description (with explanation)**

```markdown
## Summary
Fixed config loader

## Changes
- Updated config.py
```

**Problems:**

- ❌ Summary too vague - what was fixed? Why?
- ❌ Changes not specific - what was updated?
- ❌ No testing information
- ❌ Missing breaking changes section
- ❌ No checklist
- ❌ Doesn't link to issue

**Improved Version:**

```markdown
## Summary
Fixed configuration loader to properly handle environment variable substitution. Previously, undefined variables
would silently fail. Now raises ConfigError with clear message.

## Changes
- Updated `load_config()` to validate environment variables exist
- Added `ConfigError` exception for missing variables
- Added error messages with variable name and location

## Testing
- Added test for undefined variable (raises ConfigError)
- Verified error message includes variable name
- All existing tests pass

## Breaking Changes
None - only affects behavior when variables are missing (was broken before)

## Checklist
- [x] Tests added/updated
- [x] Documentation updated
- [x] Pre-commit hooks pass locally
- [x] All CI checks pass

Closes #42
```

### 4. Review Feedback Handling

#### 4.1 Two Types of Comments

**Critical Distinction:**

GitHub has TWO separate comment systems. Using the wrong one causes incomplete work.

**Type 1: PR-Level Comments**

- **Location:** Conversation tab
- **Purpose:** General updates, questions about PR as a whole
- **Command:** `gh pr comment <pr-number> --body "Comment"`
- **When to use:** Summary updates, general questions

**Type 2: Inline Review Comments**

- **Location:** Files changed tab, attached to specific code lines
- **Purpose:** Feedback on specific code
- **Command:** GitHub API (see below)
- **When to use:** Responding to code review feedback (MOST COMMON)

**See comprehensive guide:** [github-review-comments.md](/home/mvillmow/ml-odyssey-manual/agents/guides/github-review-comments.md)

#### 4.2 Responding to Inline Review Comments

**Step 1: List Review Comments**

```bash
gh api repos/OWNER/REPO/pulls/PR/comments \
  --jq '.[] | select(.user.login == "REVIEWER") | {id: .id, path: .path, body: .body}'
```

**Step 2: Make Code Changes**

- Address each comment
- Commit and push changes

**Step 3: Reply to EACH Comment**

```bash
gh api repos/OWNER/REPO/pulls/PR/comments/COMMENT_ID/replies \
  --method POST \
  -f body="✅ Fixed - [brief description]"
```

**Response Format:**

- Start with ✅ to indicate resolved
- Keep brief (1 line preferred)
- Explain WHAT was done
- Example: `✅ Fixed - Updated conftest.py to use real repository root instead of mock tmp_path`

**Step 4: Verify Replies Posted**

```bash
gh api repos/OWNER/REPO/pulls/PR/comments \
  --jq '.[] | select(.in_reply_to_id) | {replying_to: .in_reply_to_id, body: .body}'
```

**Expected:** One reply for each review comment

**Step 5: Check CI**

```bash
sleep 30  # Wait for CI to start
gh pr checks PR_NUMBER
```

**Expected:** All checks passing

#### 4.3 Verification Requirements

**Before reporting completion, MUST verify:**

1. **All comments addressed:**
   - Count review comments
   - Count your replies
   - Numbers must match

2. **Replies posted correctly:**
   - Check `in_reply_to_id` is set
   - Verify reply text is correct
   - No duplicate replies

3. **CI passing:**
   - Wait 30 seconds for CI to start
   - Check all checks pass
   - Investigate any failures

**See comprehensive verification guide:** [verification-checklist.md](/home/mvillmow/ml-odyssey-manual/agents/guides/verification-checklist.md)

#### 4.4 Re-requesting Review

**After addressing all feedback:**

1. Verify all changes committed and pushed
2. Verify CI passes
3. Verify all review comments have replies
4. Re-request review:

```bash
gh pr review PR_NUMBER --request-review REVIEWER_USERNAME
```

Or use GitHub web interface: "Re-request review" button next to reviewer name

### 5. Examples and Best Practices

#### 5.1 Common Mistakes

**Mistake 1: Using PR Comment Instead of Review Reply**

```bash
# ❌ WRONG - doesn't reply to review comments
gh pr comment 1559 --body "Fixed all review comments"

# ✅ CORRECT - reply to each review comment
gh api repos/OWNER/REPO/pulls/1559/comments/COMMENT_ID/replies \
  --method POST \
  -f body="✅ Fixed - [description]"
```

**Mistake 2: Not Verifying Replies**

```bash
# ❌ WRONG - assume it worked
gh api repos/.../comments/123/replies --method POST -f body="Fixed"
# (no verification)

# ✅ CORRECT - verify it worked
gh api repos/.../comments/123/replies --method POST -f body="Fixed"
gh api repos/.../comments --jq '.[] | select(.in_reply_to_id == 123)'
```

**Mistake 3: Checking CI Too Early**

```bash
# ❌ WRONG - CI hasn't started yet
git push
gh pr checks PR  # Shows incomplete status

# ✅ CORRECT - wait for CI to start
git push
sleep 30
gh pr checks PR
```

**Mistake 4: Vague PR Description**

```markdown
# ❌ WRONG
## Summary
Fixed stuff

# ✅ CORRECT
## Summary
Fixed configuration loader to properly validate environment variables and raise
clear errors when variables are undefined.
```

**Mistake 5: Missing Issue Link**

```bash
# ❌ WRONG - PR not linked to issue
gh pr create

# ✅ CORRECT - automatically link to issue
gh pr create --issue 123
```

#### 5.2 Best Practices

**PR Size:**

- Keep PRs focused and small (< 400 lines changed when possible)
- One issue per PR
- Split large features into multiple PRs
- Each PR should be independently reviewable

**Commit Messages:**

- Follow conventional commits: `feat:`, `fix:`, `docs:`, etc.
- Clear, concise descriptions
- Reference issue number when relevant

**Self-Review:**

- Review your own PR before requesting review
- Check diff for unintended changes
- Run tests locally
- Verify CI passes

**Communication:**

- Respond promptly to review comments
- Ask questions if feedback is unclear
- Be open to feedback
- Explain reasoning when you disagree

**Efficiency:**

- Address all feedback in one round when possible
- Batch related changes together
- Avoid force-pushing during review (hard to track changes)
- Keep PR updated with main branch

#### 5.3 Example Workflow

**Complete workflow from start to merge:**

```bash
# 1. Start from issue
gh issue view 123

# 2. Create branch
git checkout -b 123-add-feature

# 3. Write tests (TDD)
# (edit test files)
pytest

# 4. Implement feature
# (edit source files)
pytest

# 5. Update documentation
# (edit README, docstrings)

# 6. Run pre-commit
pre-commit run --all-files

# 7. Commit changes
git add .
git commit -m "feat: add new feature

Implements X to solve Y problem.

Closes #123"

# 8. Push and create PR
git push -u origin 123-add-feature
gh pr create --issue 123

# 9. Wait for review
# (reviewer leaves comments)

# 10. Address feedback
# (make changes)
git add .
git commit -m "fix: address review feedback"
git push

# 11. Reply to each review comment
gh api repos/OWNER/REPO/pulls/PR/comments/ID1/replies \
  --method POST \
  -f body="✅ Fixed - [description]"
# (repeat for each comment)

# 12. Verify replies
gh api repos/OWNER/REPO/pulls/PR/comments \
  --jq '.[] | select(.in_reply_to_id)'

# 13. Check CI
sleep 30
gh pr checks PR

# 14. Re-request review
gh pr review PR --request-review REVIEWER

# 15. After approval, merge
gh pr merge PR --squash --delete-branch
```

## Next Steps

This specification enables the following parallel phases to proceed:

1. **Issue #184 (Test):** Create test plan for documentation quality verification
2. **Issue #185 (Implementation):** Write/update PR process section in CONTRIBUTING.md
3. **Issue #186 (Packaging):** Verify integration with existing documentation
4. **Issue #187 (Cleanup):** Polish and finalize documentation

## Plan Completion Status

- [x] Analyzed existing PR process documentation
- [x] Identified gaps and areas for improvement
- [x] Designed comprehensive PR description template
- [x] Specified review process transparency improvements
- [x] Planned review feedback handling section
- [x] Created examples of good and bad practices
- [x] Defined integration with existing documentation
- [x] Established testing, implementation, packaging, and cleanup approaches
- [x] Documented all design decisions with rationales

**Planning phase complete.** Related issues can now proceed.
