# [Strategic] Evaluate Automation Framework Extraction

## Labels

`planning`, `strategic-decision`, `architecture`

## Summary

Evaluate whether to extract the automation framework (hierarchical planning + GitHub issue generation + agentic workflows) into a standalone GitHub template repository, or keep it unified with the ML implementation work. Document the decision and rationale in an Architecture Decision Record (ADR).

## Problem Statement

A reviewer observed: *"There's really two projects going on here that could each stand on their own two feet. I wonder if it would be worthwhile to fork the automated GH interactions into their own GitHub template and focus on that as a repo of its own."*

### Current state

- Unified repository mixing ML implementation goals with automation infrastructure
- Automation framework has potential standalone value:
  - 4-level hierarchical planning system
  - Automated GitHub issue generation from plan files
  - 6-level agentic workflow orchestration
  - Development tooling and scripts
- No architectural decision record documenting the unified approach
- Unclear whether automation should be extracted or remain integrated

**Key tension:** The automation framework is both:

1. **Infrastructure for ML implementation** (means to an end)
1. **Potentially valuable standalone tool** (end in itself)

## Objective

Make an informed strategic decision about repository architecture through:

1. Analysis of extraction vs unified approaches
1. Evaluation of trade-offs
1. Community input (if relevant)
1. Documentation of decision in ADR
1. Implementation path (if extraction chosen)

## Deliverables

### 1. Architecture Decision Record (ADR)

Create `notes/review/adr/ADR-002-automation-framework-architecture.md` containing:

```markdown
# ADR-002: Automation Framework Architecture Strategy

## Status
[Accepted | Rejected | Superseded]

## Context
[Background on the dual-purpose nature of the repository]

## Decision Drivers
[Key factors influencing the decision]

## Options Considered

### Option 1: Extract Automation Framework
[Detailed analysis]

### Option 2: Keep Unified Repository
[Detailed analysis]

### Option 3: Hybrid Approach
[If applicable]

## Decision
[Clear statement of chosen approach]

## Rationale
[Why this decision was made]

## Consequences
[Positive and negative outcomes]

## Implementation
[If extraction chosen, outline migration plan]
[If unified chosen, outline documentation strategy]

## References
[External feedback, similar projects, relevant research]
```text

### 2. Trade-off Analysis

Document analysis of each approach:

#### Option 1: Extract Automation Framework

### Structure:

```text
Separate repositories:
- ml-odyssey-automation (GitHub template)
  - scripts/ (issue generation)
  - .claude/agents/ (agent hierarchy)
  - .templates/ (plan templates)
  - docs/ (automation guides)

- ml-odyssey (ML implementations)
  - papers/ (paper implementations)
  - Uses ml-odyssey-automation as template
```text

### Pros:

- Clear separation of concerns
- Automation framework reusable for other projects
- Simpler for each audience (project managers vs ML researchers)
- Easier to maintain focused documentation
- Can version automation independently

### Cons:

- Maintenance overhead for two repositories
- Coordination complexity for changes affecting both
- Risk of fragmentation
- Lose dogfooding benefits (testing automation on ML work)
- Migration effort required

### Unknowns:

- Is there actual community demand for standalone automation?
- How much effort to maintain dual repos?
- Will automation and ML concerns diverge or remain coupled?

#### Option 2: Keep Unified Repository

### Structure:

```text
ml-odyssey (current structure)
- papers/ (ML implementations)
- scripts/ (automation)
- .claude/agents/ (agent hierarchy)
- notes/ (planning and documentation)
```text

### Pros:

- Simpler maintenance (single repository)
- Dogfooding automation on ML implementation
- Shared evolution of both concerns
- No migration effort needed
- Single community

### Cons:

- Mixed audiences (confusing for some users)
- Unclear primary purpose
- Harder to extract automation for reuse
- Documentation must serve dual purposes

### Mitigation strategies:

- Better README explaining dual purpose
- Clear documentation organization
- Separate guides for automation vs ML implementation
- Consider extraction in future if demand materializes

#### Option 3: Hybrid Approach

### Structure:

- Keep unified for now
- Design automation as extractable modules
- Document extraction path for future
- Publish automation guides that work standalone

### Pros:

- Preserves optionality
- Allows validation of standalone value before committing
- Lower immediate effort

### Cons:

- Requires careful architectural boundaries
- May complicate current development

### 3. Community Input (Optional)

If relevant, gather input on:

- Interest in standalone automation framework
- Use cases beyond ML implementation
- Willingness to contribute to either/both repositories

### 4. Implementation Plan

If extraction chosen, outline:

1. **Repository setup:**
   - Create ml-odyssey-automation repository
   - Configure as GitHub template
   - Set up CI/CD

1. **Content migration:**
   - Move scripts/ → ml-odyssey-automation/
   - Move .claude/agents/ → ml-odyssey-automation/.claude/agents/
   - Move relevant docs → ml-odyssey-automation/docs/
   - Create templates and examples

1. **Integration:**
   - Update ml-odyssey to use automation as template
   - Document integration process
   - Maintain compatibility

1. **Documentation updates:**
   - Update both READMEs
   - Create migration guide
   - Update agent prompts

If unified chosen, outline:

1. **Documentation strategy:**
   - Enhance README with dual-purpose explanation
   - Separate guides for automation vs ML concerns
   - Clear navigation for each audience

1. **Architectural boundaries:**
   - Keep automation modular and extractable
   - Document interfaces between concerns
   - Maintain separation where possible

## Key Questions to Answer

1. **Audience:**
   - Who is the primary audience for each concern?
   - Do these audiences overlap or diverge?

1. **Maintenance:**
   - What is the maintenance capacity for multiple repositories?
   - How often do automation changes affect ML implementation?

1. **Demand:**
   - Is there evidence of community interest in standalone automation?
   - What are the use cases beyond ML paper implementation?

1. **Timing:**
   - Is now the right time to extract, or wait until ML implementations exist?
   - Does extraction make more sense after dogfooding is complete?

1. **Value:**
   - What is the standalone value of the automation framework?
   - Does extraction increase or decrease total value?

1. **Complexity:**
   - How complex would dual-repo maintenance be?
   - What coordination overhead exists?

## Success Criteria

- [ ] All options analyzed with pros/cons documented
- [ ] Key questions answered with evidence
- [ ] Trade-offs clearly articulated
- [ ] Decision made and documented in ADR
- [ ] Rationale is defensible and clear
- [ ] Implementation path outlined (regardless of decision)
- [ ] Community input considered (if applicable)
- [ ] Blog post style maintained (conversational, data-driven, self-aware)

## Implementation Notes

### Research Required

1. **Similar projects:** How do other automation-heavy projects organize?
1. **GitHub template best practices:** What makes a good template repository?
1. **Community demand:** Any evidence of interest in project automation tools?

### Decision Criteria

Weight factors based on project goals:

- **Primary goal:** Implement ML papers in Mojo (not build automation tools)
- **Secondary goal:** Build robust automation infrastructure
- **Meta goal:** Learn about agentic workflows and LLM-assisted development

If primary goal dominates → lean toward unified (automation is means to end)
If secondary goal equally important → lean toward extraction (automation as end itself)

### Style Guide

Follow blog post style for ADR:

- Conversational but rigorous
- Data-driven where possible
- Self-aware about uncertainties
- Explain reasoning, not just conclusions

### Example tone:

> "The automation framework emerged organically while building infrastructure for ML implementations.
> The question is whether it has standalone value, or if extracting it would be premature optimization.
> Let's look at the data..."

## References

- notes/blog/11-7-2025.md (automation development journey)
- notes/blog/11-13-2025.md (dogfooding infrastructure)
- External feedback (November 14, 2025)
- notes/review/adr/ADR-001-language-selection-tooling.md (ADR template reference)

## Estimated Effort

**Medium-Large** - Strategic analysis and decision-making

- Research and analysis: 3-4 hours
- Trade-off evaluation: 2-3 hours
- ADR writing: 2-3 hours
- Community input (if applicable): 1-2 hours
- Implementation plan outline: 2-3 hours

### Total: ~10-15 hours

If extraction chosen, implementation adds significant additional effort (~20-40 hours).

## Dependencies

None - Pure strategic analysis

However, decision should consider:

- Completion of Issue #1 (plan file documentation)
- Completion of Issue #2 (README clarity)
- Current state of ML implementations (none exist yet)

## Follow-up Tasks

If extraction chosen:

- Create ml-odyssey-automation repository
- Execute migration plan
- Update documentation
- Set up cross-repo workflows

If unified chosen:

- Enhance documentation for dual-purpose repo
- Establish architectural boundaries
- Document future extraction path

Regardless of decision:

- Implement chosen approach
- Monitor decision outcomes
- Revisit if circumstances change

## Timeline

**No rush** - This is a strategic decision that benefits from deliberation.

Recommended timeline:

1. Complete Issues #1 and #2 first (improve current documentation)
1. Gather community input if relevant
1. Make decision when first ML implementation starts
1. Allows validation of automation value through dogfooding before extracting

## Risk Assessment

### Risks of extraction:

- Premature optimization (no ML implementations to validate automation yet)
- Fragmentation of effort
- Coordination overhead
- Loss of dogfooding benefits

### Risks of unified:

- Confusing dual purpose
- Harder to extract later if demand materializes
- Mixed audiences in single community

### Mitigation:

- Regardless of decision, maintain modular automation components
- Keep extraction as future option
- Document decision rationale clearly

---

**Priority**: MEDIUM - Strategic decision that benefits from deliberation and data
**Recommended timing**: After Issues #1 and #2 complete, before or during first ML implementation
