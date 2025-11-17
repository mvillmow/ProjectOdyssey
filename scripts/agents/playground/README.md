# Agent Playground Scripts

This directory contains historical and experimental scripts used during agent system development.
These scripts are kept for reference but are generally not needed for regular development.

## Status: Deprecated / Historical

**Note**: These scripts were used during initial agent system setup and refactoring.
Most functionality has been superseded by more robust solutions.

## Scripts

### Agent Refactoring Scripts (Legacy)

These scripts were used to refactor agent configuration files during initial setup:

- **`fix_duplicate_delegation.py`** - Fixed duplicate Delegation sections in agent files
- **`cleanup_agent_redundancy.py`** - Removed redundant sections from agent files
- **`fix_agent_markdown.py`** - Fixed markdown linting errors in agent files
- **`condense_pr_sections.py`** - Replaced verbose PR sections with references
- **`condense_mojo_guidelines.py`** - Condensed Mojo guidelines sections

**Status**: These were one-time refactoring scripts. Agent files have been updated and
these scripts are no longer needed for regular development.

### Testing Scripts

- **`create_single_component_issues.py`** - Create issues for a single component (testing)

**Status**: Deprecated in favor of `create_issues.py --file` option.

## Usage

These scripts should generally not be run unless you're:

1. Doing similar refactoring on a fork
2. Understanding the historical development process
3. Extracting patterns for new automation scripts

## Modern Alternatives

- **For issue creation**: Use `scripts/create_issues.py --file <path>` instead of `create_single_component_issues.py`
- **For agent markdown fixes**: Use `scripts/fix_markdown.py` for general markdown fixing
- **For agent validation**: Use `scripts/agents/validate_agents.py`

## Notes

- All scripts use shared utilities from `scripts/common.py` for path detection
- Scripts include type hints and proper documentation
- Error handling varies (these were quick refactoring tools)
