# Scripts Archive

This directory contains historical scripts that served their purpose and are no longer actively used, but are
preserved for reference.

## Purpose

Scripts are archived when they:

1. Were one-time utilities that have completed their task
2. Have been replaced by better implementations
3. Are no longer needed due to repository evolution
4. Should be preserved for historical context

## Archived Scripts

### Markdown Linting Fixes (November 2024)

During the repository's markdown standardization effort, these scripts were created to automatically fix common
markdown linting errors across the codebase.

**Scripts**:

- `fix_markdown.py` - Systematic markdown linting fixes (170 LOC)
  - Fixed MD022 (blank lines around headings)
  - Fixed MD031 (blank lines around code blocks)
  - Fixed MD032 (blank lines around lists)
  - Fixed MD040 (language tags for code blocks)
  - Fixed MD012 (multiple consecutive blank lines)

- `fix_markdown_linting.py` - Repository-wide markdown fixes (182 LOC)
  - Batch processing of all markdown files
  - Applied same fixes as fix_markdown.py
  - Hardcoded repository path for automation

- `fix_remaining_markdown.py` - Final cleanup pass (124 LOC)
  - Fixed MD040 (missing language tags)
  - Fixed MD026 (trailing punctuation in headings)
  - Fixed MD036 (bold text used as headings)
  - Fixed MD029 (ordered list numbering)
  - Targeted specific files with known issues

**Status**: These scripts successfully standardized all markdown files to pass `markdownlint-cli2` linting. The
repository now uses pre-commit hooks to maintain markdown quality, making these scripts obsolete.

**Timeline**: Created and used during markdown standardization effort (October-November 2024), archived after
successful completion.

### Historical Scripts (Not Found)

The following scripts were mentioned in Issue #9 but do not exist in the current repository, suggesting they
were already removed in previous cleanup efforts:

- `simple_update.py` - Early issue update script
- `update_tooling_issues.py` - Section-specific updater
- `update_ci_cd_issues.py` - Section-specific updater
- `update_agentic_workflows_issues.py` - Section-specific updater
- `update_github_issues.sh` - Shell-based updater
- `batch_update.sh` - Batch processing script
- `run_update.sh` - Automation wrapper

These were likely predecessors to the current `regenerate_github_issues.py` script, which consolidates all
github_issue.md regeneration into a single, robust tool.

## Active Scripts

For currently active scripts, see the main [scripts/README.md](../README.md).

**Active as of November 2024**:

- `create_issues.py` - Main GitHub issue creation tool
- `create_single_component_issues.py` - Single component testing utility
- `regenerate_github_issues.py` - Dynamic github_issue.md generation
- `agents/` - Agent system utilities (subdirectory with multiple tools)

## Using Archived Scripts

While these scripts are no longer actively maintained, they can be examined for:

- Historical context on repository evolution
- Reference implementations for similar tasks
- Understanding past automation approaches

**Warning**: Archived scripts may have hardcoded paths, outdated dependencies, or assume repository structures
that no longer exist. Use for reference only.

## Archival Policy

Scripts should be archived when:

1. Their primary purpose has been completed
2. They have been superseded by better tools
3. They are no longer compatible with current repository structure
4. They are one-time utilities that won't be run again

Scripts should NOT be archived if:

1. They are still used in active workflows
2. They are referenced in documentation as current tools
3. They are part of CI/CD pipelines
4. They have ongoing maintenance or enhancement plans

## Restoration

If an archived script is needed again:

1. Review the script for compatibility with current repository structure
2. Update any hardcoded paths or deprecated dependencies
3. Test thoroughly before moving back to active scripts
4. Update relevant documentation
5. Consider whether the functionality should be integrated into existing active tools instead
