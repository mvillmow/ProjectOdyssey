# Issue #9: Remove Duplicate Script Functionality

## Objective

Clean up the scripts directory by archiving obsolete scripts and updating documentation to reflect the current
active state of the repository's automation tools.

## Deliverables

- `scripts/archive/` directory created for historical scripts
- `scripts/archive/README.md` documenting historical context
- Three markdown fix scripts moved to archive:
  - `fix_markdown.py`
  - `fix_markdown_linting.py`
  - `fix_remaining_markdown.py`
- `scripts/README.md` updated to reflect current structure and archive
- Decision documented to keep both `create_issues.py` and `create_single_component_issues.py`

## Success Criteria

- [ ] `scripts/archive/` directory exists
- [ ] Historical markdown fix scripts moved to archive
- [ ] `scripts/archive/README.md` created with context
- [ ] `scripts/README.md` updated with archive section
- [ ] Active scripts clearly documented (create_issues.py, create_single_component_issues.py,
  regenerate_github_issues.py)
- [ ] Agent scripts in `scripts/agents/` documented
- [ ] All markdown files pass linting
- [ ] Commit created with cleanup changes

## References

- CLAUDE.md - Project documentation standards
- scripts/README.md - Scripts directory documentation
- Issue #9 - Original cleanup request

## Implementation Notes

### Current State Audit (Pre-Cleanup)

**Active Scripts** (keep in main scripts directory):

1. `create_issues.py` - Main GitHub issue creation tool (854 LOC)
2. `create_single_component_issues.py` - Testing utility for single component (198 LOC)
3. `regenerate_github_issues.py` - Dynamic github_issue.md generation (450+ LOC)

**Agent Scripts** (keep in scripts/agents/ subdirectory):

- `agent_health_check.sh` - Agent system health checks
- `agent_stats.py` - Agent statistics and metrics
- `check_frontmatter.py` - YAML frontmatter validation
- `list_agents.py` - Agent discovery and listing
- `setup_agents.sh` - Agent setup automation
- `test_agent_loading.py` - Agent loading tests
- `validate_agents.py` - Agent configuration validation

**Historical Scripts** (move to archive):

1. `fix_markdown.py` - One-time markdown linting fixes (170 LOC)
2. `fix_markdown_linting.py` - One-time markdown linting fixes (182 LOC)
3. `fix_remaining_markdown.py` - One-time markdown linting fixes (124 LOC)

### Key Findings

1. **No Historical Scripts from Issue Description Found**: The original issue mentioned scripts like
   `simple_update.py`, `update_tooling_issues.py`, `update_ci_cd_issues.py`, `update_agentic_workflows_issues.py`,
   and various `.sh` files. None of these exist in the current repository, suggesting cleanup already occurred.

2. **Markdown Fix Scripts Are Historical**: The three markdown fix scripts were one-time utilities used during
   markdown linting standardization. They served their purpose and are no longer actively needed, but should be
   preserved in archive for historical reference.

3. **Agent Scripts Are Active**: The `scripts/agents/` subdirectory contains active tooling for the agent system
   and should remain in the main scripts directory.

4. **Decision on create_single_component_issues.py**: Keep as independent testing utility. This follows Option A
   from the planning phase - the script has clear independent value for validation and testing before bulk
   operations.

### Archive Strategy

- Move only one-time markdown fix utilities to archive
- Preserve all active scripts in main directory
- Document historical context in archive README
- Update main scripts README to explain archive and current structure
