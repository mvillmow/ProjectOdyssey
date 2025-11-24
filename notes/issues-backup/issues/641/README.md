# Issue #641: [Plan] Update Gitignore - Design and Documentation

## Objective

Design and document the comprehensive update of the .gitignore file to exclude generated files, build artifacts, virtual
environments, IDE files, and ML-specific temporary files from version control.

## Deliverables

- Updated .gitignore file at repository root
- Patterns for Python, Mojo, ML artifacts
- IDE and OS-specific ignore patterns
- Comments organizing different sections

## Success Criteria

- [ ] .gitignore file is comprehensive and up-to-date
- [ ] All common generated files are ignored
- [ ] File is organized with clear sections
- [ ] No unnecessary files will be committed

## Design Decisions

### Current State Analysis

The existing .gitignore file (19 lines) includes basic patterns:

- Pixi environments (`.pixi/*` with exception for `config.toml`)
- Python cache directories (`__pycache__`)
- Build artifacts (`logs/`, `build/`, `worktrees/`, `dist/`)
- MkDocs artifacts (`site/`, `.cache/`)
- Vim swap files (`*.swp`)

### Architectural Approach

**Organization Strategy**: Section-based organization with clear comments

The .gitignore file will be organized into logical sections for maintainability:

1. **Python-specific patterns** - Virtual environments, bytecode, packaging artifacts
1. **Mojo/MAX-specific patterns** - Mojo compilation artifacts, MAX engine cache
1. **ML-specific patterns** - Model checkpoints, training logs, datasets, tensorboard files
1. **Build artifacts** - General build outputs, distribution files
1. **IDE and editor files** - VS Code, PyCharm, Vim, Emacs, etc.
1. **OS-specific files** - macOS, Windows, Linux system files
1. **Project-specific patterns** - Pixi, logs, worktrees (preserve existing patterns)

### Pattern Coverage Strategy

- **Comprehensive over minimal** - Include common patterns even if not currently used (future-proofing)
- **Preserve existing patterns** - Keep all current patterns that are working
- **Comments for clarity** - Each section clearly labeled for easy navigation
- **Specific before general** - More specific patterns listed before wildcards

### Rationale

### Why comprehensive patterns?

- ML projects generate many file types (checkpoints, logs, datasets, visualizations)
- Development involves multiple tools (IDEs, editors, OS utilities)
- Python and Mojo both generate compilation artifacts
- Better to ignore too much than accidentally commit large/sensitive files

### Why section-based organization?

- Easier to maintain and update specific categories
- Team members can quickly find relevant patterns
- Clear documentation of what is ignored and why
- Follows industry best practices (GitHub's gitignore templates)

### Why preserve existing patterns?

- Current patterns are working and part of established workflow
- Pixi environment management is critical to project
- Build artifacts pattern prevents large file commits
- No reason to break existing functionality

### Alternatives Considered

### Alternative 1: Minimal .gitignore (rejected)

- Only include patterns for files currently generated
- **Pros**: Smaller file, less maintenance
- **Cons**: Must update every time new tools are used, risk accidentally committing generated files
- **Rejection reason**: ML projects evolve rapidly, better to be comprehensive upfront

**Alternative 2: Multiple .gitignore files (rejected)**

- Use .gitignore files in subdirectories for different patterns
- **Pros**: Patterns closer to relevant code
- **Cons**: Harder to maintain, easy to miss patterns, not standard practice
- **Rejection reason**: Single root .gitignore is industry standard and easier to manage

### Alternative 3: Global .gitignore (rejected)

- Use `~/.gitignore_global` for IDE and OS patterns
- **Pros**: Cleaner repository .gitignore
- **Cons**: Requires each developer to configure, not portable, team may use different tools
- **Rejection reason**: Repository .gitignore ensures consistent behavior for all contributors

### Implementation Pattern

### File Structure

```text
# Section Header (e.g., "Python")
# Brief description if needed

pattern1
pattern2
*.extension

# Next Section Header
...
```text

### Pattern Types

- **Directory patterns**: End with `/` (e.g., `__pycache__/`)
- **Extension patterns**: Use wildcard (e.g., `*.pyc`)
- **Specific files**: Full name (e.g., `.DS_Store`)
- **Nested patterns**: Use `**/` for any directory depth (e.g., `**/*.pyc`)

### Key Technical Decisions

1. **Keep Pixi exception** - `!.pixi/config.toml` pattern preserved because Pixi config should be version-controlled
1. **Include dataset patterns** - Ignore common dataset directories (`data/`, `datasets/`) to prevent large file commits
1. **ML framework coverage** - Include patterns for PyTorch, TensorFlow, and other common ML libraries
1. **Mojo compilation artifacts** - Include `.mojo.ll`, `.mojo.o`, and other Mojo-specific build files
1. **Test coverage** - Ignore test coverage reports and artifacts (`.coverage`, `htmlcov/`, `.pytest_cache/`)

## References

- **Source Plan**: [notes/plan/01-foundation/02-configuration-files/03-git-config/01-update-gitignore/plan.md](notes/plan/01-foundation/02-configuration-files/03-git-config/01-update-gitignore/plan.md)
- **Parent Context**: [notes/plan/01-foundation/02-configuration-files/03-git-config/plan.md](notes/plan/01-foundation/02-configuration-files/03-git-config/plan.md)
- **Related Issues**:
  - Issue #642: [Test] Update Gitignore
  - Issue #643: [Implementation] Update Gitignore
  - Issue #644: [Package] Update Gitignore
  - Issue #645: [Cleanup] Update Gitignore

## Implementation Notes

(This section will be filled during implementation phase with findings, challenges, and decisions made during development)
