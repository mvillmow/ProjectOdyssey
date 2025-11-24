# Issue #9: Consolidate Markdown Script Functionality

## Objective

Clean up the scripts directory by consolidating duplicate markdown fixing scripts into a single unified,
reusable tool that provides ongoing value.

## Deliverables

- `scripts/fix_markdown.py` - Unified markdown linting fixer (replaces 3 separate scripts)
- Old markdown scripts removed (fix_markdown.py, fix_markdown_linting.py, fix_remaining_markdown.py)
- `scripts/README.md` updated with documentation for new unified script
- Decision documented to keep both `create_issues.py` and `create_single_component_issues.py`

## Success Criteria

- [x] Unified `scripts/fix_markdown.py` created with comprehensive features
- [x] Old duplicate markdown scripts removed
- [x] `scripts/README.md` updated with new script documentation
- [x] Active scripts clearly documented (create_issues.py, create_single_component_issues.py,
  regenerate_github_issues.py, fix_markdown.py)
- [x] Agent scripts in `scripts/agents/` documented
- [x] All markdown files pass linting
- [x] Unified script tested and verified (fixed 1100+ markdown linting errors in commit c704b38)
- [x] Commit created with consolidation changes (commit 7ab79ce)

## References

- CLAUDE.md - Project documentation standards
- scripts/README.md - Scripts directory documentation
- Issue #9 - Original cleanup request

## Implementation Notes

### Current State Audit (Pre-Consolidation)

**Active Scripts** (keep in main scripts directory):

1. `create_issues.py` - Main GitHub issue creation tool (854 LOC)
1. `create_single_component_issues.py` - Testing utility for single component (198 LOC)
1. `regenerate_github_issues.py` - Dynamic github_issue.md generation (450+ LOC)
1. `fix_markdown.py` - Unified markdown linting fixer (NEW - consolidates 3 scripts)

**Agent Scripts** (keep in scripts/agents/ subdirectory):

- `agent_health_check.sh` - Agent system health checks
- `agent_stats.py` - Agent statistics and metrics
- `check_frontmatter.py` - YAML frontmatter validation
- `list_agents.py` - Agent discovery and listing
- `setup_agents.sh` - Agent setup automation
- `test_agent_loading.py` - Agent loading tests
- `validate_agents.py` - Agent configuration validation

**Duplicate Scripts** (remove and replace with unified version):

1. `fix_markdown.py` (old) - Basic markdown linting fixes (170 LOC)
1. `fix_markdown_linting.py` - Repository-wide markdown fixes (182 LOC)
1. `fix_remaining_markdown.py` - Final cleanup pass (124 LOC)

### Key Findings

1. **No Historical Scripts from Issue Description Found**: The original issue mentioned scripts like
   `simple_update.py`, `update_tooling_issues.py`, `update_ci_cd_issues.py`, `update_agentic_workflows_issues.py`,
   and various `.sh` files. None of these exist in the current repository, suggesting cleanup already occurred.

1. **Markdown Fix Scripts Are Useful**: The three markdown fix scripts were originally one-time utilities but
   provide ongoing value for maintaining markdown quality. Instead of archiving, they should be consolidated
   into a single, well-designed, reusable tool.

1. **Agent Scripts Are Active**: The `scripts/agents/` subdirectory contains active tooling for the agent system
   and should remain in the main scripts directory.

1. **Decision on create_single_component_issues.py**: Keep as independent testing utility. This follows Option A
   from the planning phase - the script has clear independent value for validation and testing before bulk
   operations.

### Consolidation Strategy

**Instead of archiving, consolidate into unified tool**:

- Create new `scripts/fix_markdown.py` that combines best features of all 3 scripts
- Improvements over original scripts:
  - Command-line argument parsing (file/directory support)
  - Dry-run mode for safe testing
  - Verbose output option
  - Better error handling and reporting
  - Comprehensive fix coverage (8 markdown rules)
  - Excludes common directories automatically
- Remove old duplicate scripts
- Update scripts/README.md with comprehensive documentation
- Provide usage examples

### Rationale for Consolidation

- Markdown quality maintenance is an ongoing need (not one-time)
- Unified tool is easier to maintain than 3 separate scripts
- Better UX with proper CLI and options
- Reduces code duplication
- Provides long-term value for repository
