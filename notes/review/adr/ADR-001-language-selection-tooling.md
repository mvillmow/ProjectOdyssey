# ADR-001: Pragmatic Hybrid Language Approach for Tooling and Automation

**Status**: Accepted

**Date**: 2025-11-10

**Issue Reference**: [Issue #8](https://github.com/mvillmow/ml-odyssey/issues/8) - Mojo Script Conversion Feasibility Assessment

**Decision Owner**: Chief Architect

## Executive Summary

This ADR establishes a **Pragmatic Hybrid Language Approach** for the ML Odyssey project, resolving the conflict
between the project's "Mojo First" philosophy and the practical limitations of Mojo v0.25.7 for automation and
tooling tasks.

**Core Decision**: Use the right tool for the job. Mojo for ML/AI implementations and performance-critical code;
Python for automation tasks that require subprocess output capture, regex parsing, or GitHub API interaction.

**Strategic Rationale**: Project velocity and reliability are more important than philosophical consistency. We will
revisit Mojo conversion when the language matures to support the required capabilities.

## Context

### The Philosophical Conflict

The ML Odyssey project was established with a "Mojo First" philosophy as documented in CLAUDE.md:

> **Rule of Thumb**: If you're asking "Should I use Mojo or Python?", the answer is Mojo. Python requires justification.

This philosophy was grounded in valid technical reasons:

- Performance: 10-100x faster for ML workloads
- Type safety: Catch errors at compile time
- Memory safety: Built-in ownership and borrow checking
- Consistency: One language across the project
- Future-proof: Designed for AI/ML from the ground up

The Chief Architect guidelines reinforced this:

> **Critical**: ALL new scripts, tools, and automation MUST be written in Mojo unless there's explicit
> justification documented in the issue.

### The Technical Reality

Issue #8 conducted a comprehensive feasibility assessment to convert the project's Python automation scripts
(1,502 LOC) to Mojo. The assessment revealed critical blockers:

**CRITICAL BLOCKER - Subprocess Output Capture**:

Mojo v0.25.7's subprocess module cannot capture stdout/stderr, which is essential for the automation scripts'
core functionality:

```python
# Python (current - WORKS)
result = subprocess.run(['gh', 'issue', 'create', ...], capture_output=True)
issue_url = result.stdout.strip()  # Get the issue URL

# Mojo v0.25.7 (DOESN'T WORK)
result = run("gh issue create ...")
# result.stdout - NOT AVAILABLE
# result.exit_code - NOT AVAILABLE
```

Without the ability to capture GitHub CLI output, the entire issue creation workflow breaks. This single
limitation makes conversion impossible.

**HIGH RISK - Regex Missing**:

The scripts use 15+ regex patterns for markdown parsing, which would require complete rewriting with manual string parsing:

```python
# Python regex patterns used throughout scripts
ISSUE_SECTION_PATTERN = re.compile(
    r'^(?:##\s+|\*\*)(Plan|Test|Implementation)...',
    re.MULTILINE
)
title_match = re.search(r'\*\*Title\*\*:\s*`([^`]+)`', content)
labels = re.findall(r'`([^`]+)`', labels_text)
```

**The Conflict**: The philosophy demands Mojo for all scripts, but Mojo cannot perform the required tasks.

### Scripts Affected

1. **create_issues.py** (854 LOC)
   - Automates GitHub issue creation via `gh` CLI
   - Requires: stdout capture (issue URLs), exit code checking, retry logic
   - Usage: 20+ subprocess calls with output capture
   - Critical to project workflow

2. **regenerate_github_issues.py** (446 LOC)
   - Generates github_issue.md from plan.md sources
   - Requires: 15+ regex patterns for markdown parsing
   - Critical for maintaining planning hierarchy

3. **create_single_component_issues.py** (197 LOC)
   - Testing utility for issue creation
   - Same requirements as create_issues.py

**Total Impact**: 1,502 lines of production automation code

### Assessment Results

Issue #8's comprehensive testing revealed:

| Capability | Available | Maturity | Risk | Blocks Conversion |
|-----------|-----------|----------|------|-------------------|
| File I/O | Yes | Mature | Low | No |
| Subprocess Exec | Yes | Alpha | Critical | No |
| **Subprocess Capture** | **No** | **Missing** | **Critical** | **YES** |
| **Exit Code Access** | **No** | **Missing** | **Critical** | **YES** |
| String Basics | Yes | Mature | Low | No |
| String Methods | Partial | Beta | Medium | No |
| **Regex** | **No** | **Missing** | **High** | No* |
| JSON | Yes | Beta | Medium | No* |
| Error Handling | Basic | Beta | Medium | No |

**\*** = Can work around but significantly increases complexity and risk

**Estimated Conversion Effort**: 7-9 weeks with low confidence

**ROI Analysis**:

- Estimated Effort: 7-9 weeks
- Estimated Benefit: Zero (current scripts work perfectly)
- Risk: High (introducing bugs into working system)
- **ROI**: Highly negative

### Decision Drivers

1. **Project Velocity**: Converting working scripts would delay ML implementation by 2-3 months
2. **Risk Management**: High probability of introducing bugs into critical automation
3. **Resource Efficiency**: 7-9 weeks of effort for zero functional benefit
4. **Technical Maturity**: Mojo v0.25.7 is not ready for systems scripting
5. **Pragmatic Engineering**: Use the right tool for each job

## Decision

### Strategic Language Boundaries

We adopt a **Pragmatic Hybrid Approach** with clear boundaries for when to use each language:

#### Mojo REQUIRED

**Core ML/AI Implementation** (Performance-Critical):

- ✅ Neural network implementations (forward/backward passes)
- ✅ Training loops and optimization algorithms
- ✅ Tensor operations and data structures
- ✅ SIMD-optimized kernels
- ✅ Performance-critical data pipelines
- ✅ Custom layer implementations
- ✅ Gradient computation
- ✅ Model inference engines

**New Code Default**:

- ✅ Any new ML algorithm implementation
- ✅ Any new performance-critical component
- ✅ Any new data processing requiring SIMD
- ✅ Any code where type safety is critical

#### Python ALLOWED

**Automation Requiring Subprocess Output** (Technical Limitation):

- ✅ Scripts calling `gh` CLI that need to capture output
- ✅ Scripts calling external tools requiring stdout/stderr access
- ✅ Scripts requiring command exit code checking
- ✅ CI/CD automation requiring process output

**Regex-Heavy Text Processing** (Technical Limitation):

- ✅ Markdown parsing with complex patterns
- ✅ Configuration file parsing requiring regex
- ✅ Log analysis requiring pattern matching
- ✅ Code generation requiring template substitution

**GitHub API Interaction** (Ecosystem Limitation):

- ✅ Scripts using GitHub CLI (`gh`) for issue/PR management
- ✅ Scripts using GitHub REST API via Python libraries
- ✅ Workflow automation requiring GitHub integration

**Rapid Prototyping** (With Conversion Plan):

- ⚠️ Quick validation scripts (must document Mojo conversion plan)
- ⚠️ One-off debugging utilities (mark as temporary)
- ⚠️ Experimental features (plan migration path)

#### Decision Process

When creating a new component:

1. **What is it?**
   - ML/AI implementation → Mojo (required)
   - Automation/tooling → Check requirements

2. **Does it need subprocess output capture?**
   - Yes → Python (allowed, document why)
   - No → Continue to next check

3. **Does it need regex parsing?**
   - Yes → Python (allowed, document why)
   - No → Continue to next check

4. **Does it interface with Python-only libraries?**
   - Yes → Python (allowed, document why)
   - No → **Mojo (required)**

5. **Document the decision** in code comments and issue notes

### Justification Requirements

**For Python Usage in Automation**:

All Python automation scripts MUST include a header comment explaining the justification:

```python
#!/usr/bin/env python3

"""
Script: create_issues.py
Purpose: Automate GitHub issue creation via gh CLI

Language: Python
Justification:
  - Requires subprocess stdout capture to get issue URLs from gh CLI
  - Mojo v0.25.7 subprocess module cannot capture output (blocking limitation)
  - Uses 15+ regex patterns for markdown parsing (no Mojo regex support)

Conversion Plan:
  - Monitor Mojo releases for subprocess output capture support
  - Reassess quarterly (see ADR-001 monitoring strategy)
  - Target conversion: Q2-Q3 2026 (estimated)

Reference: ADR-001, Issue #8
"""
```

**For New Python Code**:

Any new Python code MUST have an associated GitHub issue documenting:

- Why Python is required (technical limitation, ecosystem, etc.)
- What Mojo capabilities are blocking conversion
- When the decision will be reassessed
- Link to this ADR

### Update to CLAUDE.md

The "Language Preference" section in CLAUDE.md will be updated to reflect this decision:

```markdown
### Language Preference

#### Mojo First - With Pragmatic Exceptions

**Default to Mojo** for ALL new ML/AI code:

- ✅ Neural network implementations
- ✅ Training loops and optimization
- ✅ Tensor operations and SIMD kernels
- ✅ Performance-critical data processing
- ✅ Type-safe model components

**Use Python for Automation** when technical limitations require it:

- ✅ Subprocess output capture (Mojo limitation in v0.25.7)
- ✅ Regex-heavy text processing (no Mojo regex support)
- ✅ GitHub API interaction via Python libraries
- ⚠️ Must document justification (see ADR-001)

**Rule of Thumb**:

- ML/AI implementation? → Mojo (required)
- Automation needing subprocess output? → Python (allowed)
- Automation needing regex? → Python (allowed)
- Everything else? → Mojo (default)

**See**: ADR-001 for complete language selection strategy
```

### Update to Chief Architect Guidelines

The "Language Selection Strategy" section in chief-architect.md will be updated:

```markdown
### Language Selection Strategy

**Mojo Required**:

- ALL ML/AI implementations (neural networks, training, inference)
- ALL performance-critical code (SIMD kernels, tensor ops)
- ALL new code unless technical limitation documented

**Python Allowed**:

- Automation requiring subprocess output capture (Mojo v0.25.7 limitation)
- Text processing requiring regex (no Mojo stdlib support)
- GitHub API interaction via Python libraries
- Must document justification per ADR-001

**Decision Authority**:

- Chief Architect approves new Python automation
- All Python usage must link to this ADR or have issue documenting justification
- Quarterly reviews of Python code for conversion opportunities

**See**: ADR-001 for monitoring strategy and reassessment criteria
```

## Rationale

### Why Hybrid Over Pure Mojo

**Technical Feasibility**:

- Mojo v0.25.7 cannot capture subprocess output (confirmed by Issue #8 testing)
- No Mojo stdlib regex support (confirmed by documentation review)
- Workarounds (Python interop) defeat the purpose of conversion
- Manual parsing alternatives are high-risk and time-consuming

**Engineering Economics**:

- 7-9 weeks conversion effort vs. 0 functional benefit
- High risk of introducing bugs in critical automation
- Delays ML implementation (the actual project goal)
- Python scripts are battle-tested and reliable

**Strategic Focus**:

- Project goal is ML research implementation, not language purity
- Mojo's value is in ML performance, not scripting convenience
- Python excels at automation and tooling
- Use each language where it provides the most value

### Why This Decision Is Strategic, Not Tactical

This is an **architectural decision** affecting the entire project:

1. **Defines Language Boundaries**: Clear rules for when to use each language
2. **Establishes Patterns**: Template for future technology decisions
3. **Prioritizes Project Goals**: ML implementation over tooling consistency
4. **Plans for Future**: Monitoring and reassessment strategy
5. **Balances Pragmatism with Vision**: Keeps Mojo focus where it matters

This is not a one-off exception—it's a strategic framework for technology selection.

## Consequences

### Positive Consequences

**Immediate Benefits**:

- ✅ Project can proceed without 2-3 month delay
- ✅ Reliable automation scripts remain stable
- ✅ Team can focus on ML implementation (actual project goal)
- ✅ Risk of bugs in critical tooling eliminated
- ✅ Clear decision framework for future choices

**Long-Term Benefits**:

- ✅ Pragmatic approach attracts contributors
- ✅ Flexibility to adapt as Mojo matures
- ✅ Each language used for its strengths
- ✅ Quarterly reviews ensure continuous improvement
- ✅ Documented rationale for all decisions

**Strategic Benefits**:

- ✅ Demonstrates mature engineering judgment
- ✅ Prioritizes project outcomes over ideology
- ✅ Establishes pattern for future technology decisions
- ✅ Shows understanding of tool limitations

### Negative Consequences

**Technical Debt**:

- ⚠️ Two languages to maintain (Python + Mojo)
- ⚠️ Potential for inconsistent patterns across languages
- ⚠️ Learning curve for contributors
- ⚠️ Future conversion work when Mojo matures

**Mitigation Strategy**:

- Clear boundaries prevent confusion
- Quarterly reviews minimize conversion effort
- Documentation ensures consistency
- Monitoring strategy tracks Mojo progress

**Philosophical Inconsistency**:

- ⚠️ Appears to violate "Mojo First" principle
- ⚠️ May confuse contributors about language choice

**Mitigation Strategy**:

- Update documentation to clarify boundaries
- ADR provides clear justification
- Decision process guides contributors
- "Mojo First for ML" is the actual principle

**Monitoring Overhead**:

- ⚠️ Quarterly reviews require time
- ⚠️ Tracking Mojo releases for features

**Mitigation Strategy**:

- Lightweight monitoring (check changelog)
- Only test when relevant features added
- Community forums provide early signals

### Trade-offs Accepted

We explicitly accept these trade-offs:

1. **Two languages** instead of one → But each used appropriately
2. **Future conversion work** → But only when Mojo is ready
3. **Philosophical inconsistency** → But pragmatic and data-driven
4. **Monitoring overhead** → But ensures timely conversion

These trade-offs are **preferable** to:

- 2-3 month delay in ML implementation
- High risk of bugs in critical automation
- Wasted effort on impossible conversion
- Ideological purity at expense of project success

## Monitoring and Reassessment

### Quarterly Monitoring Strategy

**Schedule**: February, May, August, November (starting Q1 2026)

**Monitoring Actions**:

1. **Check Mojo Changelog**:
   - Review official release notes for subprocess improvements
   - Check for regex or pattern matching additions
   - Note any stdlib additions relevant to scripting

2. **Test Critical Capabilities**:
   - If subprocess improvements noted, test output capture
   - If regex added, test against script requirements
   - Document test results in monitoring log

3. **Community Signals**:
   - Monitor Mojo forums for scripting examples
   - Check if community is using Mojo for automation
   - Review any published Mojo tooling scripts

4. **Decision Review**:
   - If capabilities available → Plan conversion
   - If still blocked → Continue monitoring
   - Document decision in quarterly log

**Monitoring Log Location**: `/notes/review/adr/ADR-001-monitoring.md`

### Reassessment Triggers

**Must-Have Capabilities** (blocking conversion):

1. **Subprocess Output Capture**:
   - Can capture stdout and stderr
   - Can access exit codes
   - Can set timeouts
   - Can handle errors properly

2. **Regex or Equivalent**:
   - Native regex module in stdlib, OR
   - Mature pattern matching syntax, OR
   - Documented alternative parsing approach

3. **Stable JSON Module**:
   - Well-documented API
   - Community adoption
   - Battle-tested in production

**Nice-to-Have Capabilities** (improve conversion):

- Common string methods (strip, split, replace)
- Enhanced error handling
- Dataclass-like syntax
- Better documentation with examples

**Reassessment Timeline**:

- **Optimistic**: Q2 2026 (6 months from decision)
- **Realistic**: Q3 2026 (9 months from decision)
- **Conservative**: Q1 2027 (15 months from decision)

**Conversion Decision Criteria**:

When all must-have capabilities are available:

1. Create conversion plan (estimate effort)
2. Prioritize scripts by usage frequency
3. Convert high-priority scripts first
4. Validate against Python versions
5. Document lessons learned
6. Update this ADR to "Superseded"

### Conversion Priority

When Mojo capabilities are available, convert in this order:

**High Priority** (convert first):

1. `create_issues.py` - Most frequently used
2. `regenerate_github_issues.py` - Second most used
3. `create_single_component_issues.py` - Testing utility

**Evaluation Criteria**:

- Usage frequency (daily vs. occasional)
- Performance benefit from Mojo
- Complexity of conversion
- Risk of bugs during conversion

**Estimated Conversion Effort** (when Mojo ready):

- Phase 1: Capability validation (1 week)
- Phase 2: High-priority conversion (2-3 weeks)
- Phase 3: Testing and validation (1 week)
- **Total**: 4-5 weeks (vs. 7-9 weeks today)

## Alternatives Considered

### Alternative 1: Pure Mojo with Python Interop

**Approach**: Write scripts in Mojo, call Python subprocess module via interop

**Pros**:

- Maintains "Mojo First" principle
- Learning experience for team
- Could work technically

**Cons**:

- Defeats the purpose (still dependent on Python)
- Adds complexity (Mojo + Python interop layer)
- No performance benefit (bottlenecked by Python)
- More brittle (interop can break)
- Harder to maintain

**Why Rejected**: Adds complexity without benefits. If we're calling Python anyway, use Python directly.

### Alternative 2: Wait for Mojo to Mature

**Approach**: Pause all automation work until Mojo supports required capabilities

**Pros**:

- Eventually achieves "Mojo First" goal
- No technical debt
- One language eventually

**Cons**:

- 6-12 month delay (unacceptable)
- Project cannot proceed
- ML implementation blocked
- No guarantee of timeline
- Existing Python scripts sit unused

**Why Rejected**: Unacceptable project delay. Current scripts work perfectly.

### Alternative 3: Partial Conversion

**Approach**: Convert simple scripts, keep complex ones in Python

**Pros**:

- Learning experience
- Some progress toward Mojo
- Proves feasibility for simple cases

**Cons**:

- Doesn't solve the main problems
- Wastes time on non-critical work
- Same limitations apply to complex scripts
- Creates inconsistency
- No clear benefit

**Why Rejected**: Wastes effort on low-value work. Focus on ML implementation instead.

### Alternative 4: Abandon Python Scripts, Build New Mojo Workflow

**Approach**: Design new workflow that works within Mojo limitations

**Pros**:

- Pure Mojo solution
- Opportunity to redesign workflow
- No Python dependency

**Cons**:

- Requires redesigning GitHub integration
- No way to get issue URLs without subprocess output
- Would need to manually track issues
- Massive engineering effort
- Lower reliability than current system

**Why Rejected**: Infeasible without subprocess output. Can't integrate with GitHub CLI.

### Alternative 5: Hybrid Approach (SELECTED)

**Approach**: Python for automation, Mojo for ML/AI implementation

**Pros**:

- No project delay
- Uses each language appropriately
- Maintains working automation
- Allows focus on ML implementation
- Pragmatic and data-driven
- Clear decision framework

**Cons**:

- Two languages to maintain
- Future conversion work
- Philosophical inconsistency

**Why Selected**: Best balance of pragmatism, project velocity, and strategic focus. Allows immediate progress
while planning for future.

## Implementation Plan

### Phase 1: Documentation Updates (Week 1)

**Update CLAUDE.md**:

- [ ] Revise "Language Preference" section
- [ ] Add link to ADR-001
- [ ] Update examples to show both Mojo and Python cases
- [ ] Clarify decision process

**Update Chief Architect Guidelines**:

- [ ] Revise "Language Selection Strategy" section
- [ ] Add decision authority for Python usage
- [ ] Link to ADR-001
- [ ] Add quarterly review responsibility

**Create Monitoring Document**:

- [ ] Create `/notes/review/adr/ADR-001-monitoring.md`
- [ ] Document quarterly review template
- [ ] Set up tracking for Mojo releases
- [ ] Define reassessment criteria

**Update Issue #8**:

- [ ] Link to ADR-001
- [ ] Document decision accepted
- [ ] Close issue as "Decision Made"

### Phase 2: Script Documentation (Week 1-2)

**Add Justification Headers**:

- [ ] Update `create_issues.py` with header comment
- [ ] Update `regenerate_github_issues.py` with header comment
- [ ] Update `create_single_component_issues.py` with header comment
- [ ] Reference ADR-001 in all headers

**Document Current State**:

- [ ] List all Python automation scripts
- [ ] Document why each requires Python
- [ ] Identify conversion blockers for each
- [ ] Estimate conversion effort when ready

### Phase 3: Process Integration (Week 2)

**Update Agent Guidelines**:

- [ ] Update all orchestrators with language selection guidance
- [ ] Add ADR-001 reference to implementation specialists
- [ ] Update review specialists with dual-language checks

**Create Decision Template**:

- [ ] Template for justifying Python usage
- [ ] Checklist for language selection
- [ ] Process for quarterly reviews

**Team Communication**:

- [ ] Update team documentation
- [ ] Add FAQ about language selection
- [ ] Document escalation path for exceptions

### Phase 4: First Quarterly Review (Q1 2026)

**Establish Monitoring Routine**:

- [ ] Check Mojo v0.26+ changelog
- [ ] Test subprocess improvements if any
- [ ] Document findings
- [ ] Update conversion timeline if applicable

### Success Criteria

This implementation is successful when:

- [ ] All documentation updated and consistent
- [ ] All Python scripts have justification headers
- [ ] Team understands decision framework
- [ ] Quarterly monitoring established
- [ ] No confusion about when to use which language
- [ ] ML implementation proceeds without delay

## References

### Issue #8 Findings

**Comprehensive Assessment**: [Issue #8: Mojo Script Conversion Feasibility Assessment](https://github.com/mvillmow/ml-odyssey/issues/8)

**Key Findings**:

- Subprocess output capture: NOT AVAILABLE (blocking)
- Exit code access: NOT AVAILABLE (blocking)
- Regex support: NOT AVAILABLE (high risk)
- Estimated effort: 7-9 weeks (low confidence)
- ROI: Highly negative
- Recommendation: NO-GO

**Test Results**:

- File I/O: PASS (mature)
- Subprocess execution: PARTIAL (no output capture)
- String operations: PARTIAL (basic only)
- Pattern matching: MISSING (no regex)

**Decision Criteria Met**: 4 out of 4 NO-GO criteria

### Related Documentation

**Project Guidelines**:

- [CLAUDE.md](../../../CLAUDE.md) - Project-wide guidelines
- [Chief Architect Guidelines](../../../.claude/agents/chief-architect.md) - Strategic decisions

**Mojo Documentation**:

- [Mojo Changelog](https://docs.modular.com/mojo/changelog/) - Release notes
- [Mojo Stdlib](https://docs.modular.com/mojo/stdlib/) - Standard library
- [Mojo Subprocess](https://docs.modular.com/mojo/stdlib/subprocess/) - Subprocess module

**Scripts Affected**:

- `scripts/create_issues.py` - GitHub issue creation (854 LOC)
- `scripts/regenerate_github_issues.py` - Issue generation (446 LOC)
- `scripts/create_single_component_issues.py` - Testing utility (197 LOC)

### Stakeholder Communication

**Internal**:

- All section orchestrators
- Implementation specialists
- Review specialists

**External**:

- Mojo team (monitoring for feature requests)
- Contributors (via updated guidelines)

## Appendices

### Appendix A: Mojo Maturity Assessment

Based on Issue #8 testing with Mojo v0.25.7:

**Mature for Production**:

- File I/O (read/write)
- Basic string operations
- Compile-time optimization
- SIMD operations
- Memory management

**Not Ready for Production**:

- Subprocess output capture (missing)
- Exit code access (missing)
- Regex support (missing)
- Advanced string methods (partial)
- Error handling (basic)

**Estimated Timeline for Scripting Maturity**: Q2-Q3 2026

### Appendix B: Language Selection Decision Tree

```text
┌─────────────────────────────────────┐
│ What type of component?             │
└─────────────┬───────────────────────┘
              │
      ┌───────┴────────┐
      │                │
      ▼                ▼
  ML/AI Code?     Automation?
      │                │
      │                └──────┐
      ▼                       │
   USE MOJO             ┌─────┴──────┐
  (REQUIRED)            │            │
                        ▼            ▼
                 Need subprocess  Need regex?
                    output?         │
                        │            │
                  ┌─────┴─────┐     │
                  │           │     │
                  ▼           ▼     ▼
                YES          NO    YES
                  │           │     │
                  ▼           │     │
            USE PYTHON   ┌────┴─────┘
            (ALLOWED)    │
                         ▼
                    USE MOJO
                   (DEFAULT)

Document all Python usage with justification!
```

### Appendix C: Quarterly Review Template

```markdown
# ADR-001 Quarterly Review: [Quarter] [Year]

**Review Date**: YYYY-MM-DD
**Mojo Version Checked**: X.XX.X
**Reviewer**: [Name/Role]

## Changelog Review

**Subprocess Improvements**:

- [ ] Checked changelog: [Link]
- [ ] Findings: [None / Description]
- [ ] Tested: [Yes/No]

**Regex Support**:

- [ ] Checked changelog: [Link]
- [ ] Findings: [None / Description]
- [ ] Tested: [Yes/No]

**Other Relevant Changes**:

- [List any other improvements]

## Capability Testing

**If improvements noted, test results**:

- [ ] Subprocess output capture: [PASS/FAIL]
- [ ] Exit code access: [PASS/FAIL]
- [ ] Regex or alternative: [PASS/FAIL]

## Decision

- [ ] Capabilities available → Proceed to conversion planning
- [ ] Still blocked → Continue monitoring
- [ ] Partial progress → Re-evaluate at next quarter

**Next Steps**: [Description]

**Next Review**: [Quarter Year]
```

### Appendix D: Justification Header Template

```python
#!/usr/bin/env python3

"""
Script: [script_name].py
Purpose: [Brief description of what the script does]

Language: Python
Justification:
  - [Specific Mojo limitation #1]
  - [Specific Mojo limitation #2]
  - [Any other technical requirements]

Conversion Blockers:
  - [Feature 1] - Mojo version needed: [X.XX+]
  - [Feature 2] - Mojo version needed: [X.XX+]

Conversion Plan:
  - Monitor Mojo releases for [specific features]
  - Reassess quarterly (see ADR-001 monitoring strategy)
  - Target conversion: [Quarter Year] (estimated)

Reference: ADR-001, Issue #[number]
Last Review: [YYYY-MM-DD]
"""
```

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-10 | Chief Architect | Initial ADR creation |

---

**Document Metadata**:

- Location: `/notes/review/adr/ADR-001-language-selection-tooling.md`
- Status: Accepted
- Review Frequency: Quarterly
- Next Review: Q1 2026 (February 2026)
- Supersedes: Original "Mojo First" language preference in CLAUDE.md
- Superseded By: None (current)

---

*This ADR represents a strategic architectural decision affecting the entire ML Odyssey project. All changes to
language selection strategy must reference this document or create a superseding ADR.*
