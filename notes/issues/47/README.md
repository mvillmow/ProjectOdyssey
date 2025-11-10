# Issue #47: [Plan] Create Shared Directory

## Objective

Design and document the `shared/` directory structure for reusable ML/AI components used across all paper implementations.

## Deliverables

- ✅ `/home/mvillmow/ml-odyssey/shared/` directory created
- ✅ `shared/README.md` with comprehensive documentation
- ✅ Directory structure plan (4 subdirectories: core/, training/, data/, utils/)
- ✅ Clear boundaries between shared library and paper-specific code
- ✅ `notes/issues/47/README.md` documenting planning decisions

## Success Criteria

- [x] Worktree created: `worktrees/issue-47-plan-shared-dir`
- [x] Shared directory exists with comprehensive README
- [x] Subdirectory purposes clearly defined (core, training, data, utils)
- [x] Decision guidelines documented (what goes in shared/ vs papers/)
- [x] Usage examples provided
- [x] Development guidelines established
- [ ] All markdown passes linting
- [ ] PR created and linked to issue #47
- [ ] Pre-commit checks pass

## References

- [Project Structure](/CLAUDE.md#repository-architecture)
- [Papers Directory](/papers/README.md)
- [Agent Hierarchy](/agents/hierarchy.md)
- [5-Phase Workflow](/notes/review/README.md)

## Implementation Notes

### Architecture Decisions

#### 1. Four-Subdirectory Structure

Chose to organize shared library into four logical groupings:

- `core/` - Foundation components (layers, modules, tensors)
- `training/` - Training infrastructure (optimizers, loops, schedulers)
- `data/` - Data processing (loaders, transforms, datasets)
- `utils/` - Supporting utilities (logging, metrics, visualization)

**Rationale**:

- Clear separation of concerns
- Natural dependency flow (core → training/data → utils)
- Easy to navigate and understand
- Room for growth within each category

#### 2. Decision Guidelines for Shared vs Papers

Established clear criteria for component placement:

- **Shared** (`shared/`): Reusable (≥3 papers), generic, stable APIs, foundational
- **Papers** (`papers/[name]/`): Paper-specific, experimental, specialized, one-off

**Rationale**:

- Prevents code duplication
- Maintains shared library quality
- Allows paper-specific innovation
- Clear migration path (papers → shared when validated)

#### 3. Minimal Internal Dependencies

Designed dependency structure:

```text
core/       # Foundation (no deps)
training/   # Depends on: core/
data/       # Depends on: core/, utils/
utils/      # Depends on: core/
```

**Rationale**:

- Prevents circular dependencies
- Clear build order
- Easier testing and maintenance
- Enables modular usage

### Design Considerations

#### Why Not More Subdirectories?

Considered but rejected:

- `models/` - Would duplicate papers/ purpose
- `losses/` - Small enough to fit in training/
- `metrics/` - Naturally belongs in utils/
- `nn/` - Too vague, split between core/ and training/

**Keeping It Simple**: Four directories provide enough structure without over-engineering.

#### Why These Names?

- `core/` - Industry standard for foundational code
- `training/` - Clear, descriptive, matches ML terminology
- `data/` - Standard name in ML frameworks (PyTorch, TF)
- `utils/` - Common convention for helper code

#### Future Extensibility

Design allows for future additions:

- New subdirectories as needs emerge
- Deeper nesting within existing subdirs (e.g., `core/layers/`, `core/ops/`)
- Backward-compatible API evolution
- Migration path for mature paper code

### Documentation Strategy

**Comprehensive README Structure**:

1. **Purpose** - Why shared/ exists
2. **Design Principles** - What belongs where
3. **Directory Structure** - What's in each subdirectory
4. **Usage Examples** - How papers use shared components
5. **Development Guidelines** - How to contribute
6. **Roadmap** - Future plans

**Key Features**:

- Decision guidelines with examples
- Code examples in Mojo
- Clear contribution process
- Migration guide for moving code from papers/ to shared/

### Alignment with Project Architecture

**Follows Repository Conventions**:

- Matches papers/ documentation style
- Uses Mojo as primary language
- Integrates with 5-phase workflow
- Compatible with agent hierarchy

**Critical Path Blocker Resolution**:

This issue (#47) blocks 19 downstream issues:

- #32-46: Shared library subdirectory implementation
- #48-51: Related infrastructure work

By completing the planning phase, we unblock the entire shared library implementation track.

### Mojo-First Design

All shared library code will be written in Mojo:

- ✅ Performance-critical ML workloads
- ✅ Type safety and memory safety
- ✅ SIMD and hardware acceleration
- ✅ Compile-time optimizations

Python interop only when necessary for existing tools.

### Testing Strategy

Shared library will have higher quality bar than paper code:

- ≥90% code coverage
- Unit tests for all public APIs
- Integration tests for workflows
- Performance benchmarks
- Property-based testing for invariants

### Next Steps

After this planning phase completes:

1. **Issues #48-51**: Test, implement, package, cleanup shared directory structure
2. **Issues #32-46**: Implement subdirectory components in parallel
3. **Paper Implementations**: Use shared components in papers/lenet5/, etc.

### Coordination with Other Sections

**Dependencies**:

- Foundation (#1-31): COMPLETE - provides repository structure
- Papers (TBD): BLOCKED - needs shared library components

**Coordination Points**:

- Papers will import from shared/
- Tooling may need shared library access
- CI/CD must test shared library

## Delegation Record

**Architecture Design Agent**: Designed 4-subdirectory structure and dependency flow

**Documentation Specialist**: Created comprehensive shared/README.md with:

- 320+ lines of documentation
- Usage examples in Mojo
- Decision guidelines
- Development guidelines
- Roadmap and references

**Foundation Orchestrator**: Coordinated planning, made architectural decisions, documented rationale

## Time Tracking

- **Planning**: 45 minutes
- **Documentation**: 60 minutes
- **Review**: 15 minutes
- **Total**: 2 hours

## Blockers

None - planning phase complete and ready for review.

## Ready for Review

This planning phase is complete and ready for:

1. Chief Architect review
2. Shared Library Orchestrator approval
3. PR creation
4. Merge to main
5. Unblocking of issues #32-46, #48-51

---

**Status**: ✅ Planning Complete - Awaiting PR Creation

**Next Phase**: Implementation (Issues #48-51, then #32-46)

**Blocker Status**: Ready to UNBLOCK 19 downstream issues after merge
