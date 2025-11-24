# Issue #616: [Plan] Magic TOML - Design and Documentation

## Objective

Create comprehensive design documentation and specifications for the magic.toml configuration file that will define project metadata, dependencies, channels, and environment configuration for the Mojo/MAX development environment.

## Deliverables

- Detailed specification for magic.toml structure and content
- Project metadata design (name, version, description)
- Dependencies architecture (Mojo/MAX, Python packages)
- Channel configuration strategy (conda-forge, Modular channels)
- Documentation of design decisions and rationale

## Success Criteria

- [ ] Complete specification of magic.toml file structure documented
- [ ] Project metadata design finalized and documented
- [ ] Dependency selection rationale documented
- [ ] Channel configuration strategy documented
- [ ] All design decisions have clear rationale
- [ ] Implementation guide ready for Test and Implementation phases

## Design Decisions

### 1. File Structure and Organization

**Decision**: Use standard TOML format with three main sections: project metadata, dependencies, and channels.

### Rationale

- Magic package manager follows conda/pixi conventions
- Clear separation of concerns makes configuration maintainable
- TOML format is human-readable and well-supported

### Alternatives Considered

- Single flat configuration - rejected due to lack of organization
- YAML format - rejected as Magic specifically requires TOML

### 2. Project Metadata Strategy

**Decision**: Include minimal but complete metadata: name, version, description, and basic project information.

### Rationale

- Follows Magic best practices for package identification
- Provides enough information for package discovery
- Keeps configuration simple and maintainable (KISS principle)
- Avoids over-specification (YAGNI principle)

### Key Metadata Fields

- `name`: "ml-odyssey" (project identifier)
- `version`: "0.1.0" (semantic versioning, initial development)
- `description`: Clear explanation of project purpose
- `authors`: Project maintainer information
- `license`: Open source license specification

### 3. Dependency Management Architecture

**Decision**: Organize dependencies into logical groups with explicit version constraints only where necessary.

### Rationale

- Mojo/MAX require specific version compatibility
- Python packages can use more flexible versioning
- Explicit documentation of why each dependency is needed
- Minimal constraints allow easier updates (avoid dependency hell)

### Dependency Categories

1. **Core Mojo/MAX**:
   - MAX package (version constraint based on compatibility)
   - Mojo compiler (if separate package)

1. **Python Runtime**:
   - Python interpreter (version 3.9+ for MAX compatibility)

1. **Development Tools**:
   - Testing frameworks (deferred to later issues)
   - Documentation tools (deferred to later issues)

### Version Constraint Strategy

- Use `>=` for flexible dependencies (e.g., `python >= 3.9`)
- Use `~=` for compatible version ranges when needed (e.g., `max ~= 24.5`)
- Use `==` only for critical compatibility requirements
- Document all version constraints with rationale comments

### 4. Channel Configuration Strategy

**Decision**: Configure standard conda-forge channel plus Modular-specific channels with appropriate priority.

### Rationale

- conda-forge provides broad package ecosystem
- Modular channels required for MAX/Mojo packages
- Channel priority prevents package conflicts
- Standard approach familiar to conda/pixi users

### Channel Configuration

1. **Primary Channels**:
   - `conda-forge`: Standard package repository
   - Modular channel: For MAX/Mojo packages (exact URL TBD during implementation)

1. **Channel Priority**:
   - Modular channels take precedence for MAX/Mojo packages
   - conda-forge for everything else
   - Explicit priority prevents ambiguity

### 5. Documentation and Comments Strategy

**Decision**: Include inline comments explaining non-obvious choices and constraints.

### Rationale

- Future maintainers need context for decisions
- Version constraints need justification
- Channel choices may not be obvious to new contributors
- Self-documenting configuration reduces support burden

### Comment Guidelines

- Explain WHY, not WHAT (code shows what)
- Document version constraint rationale
- Note any workarounds or compatibility issues
- Reference external documentation where helpful

### 6. Minimal Configuration Principle

**Decision**: Start with minimal valid configuration, expand only as needed.

### Rationale

- YAGNI principle - don't add until required
- Easier to add than remove
- Reduces maintenance burden
- Faster iteration during development

### What to Exclude Initially

- Optional dependencies (add when needed)
- Development-only tools (separate environment)
- Platform-specific configurations (add if needed)
- Advanced features not required for basic setup

## Implementation Components

Based on the design decisions above, the implementation will proceed in three phases (handled by separate issues):

### Phase 1: Base Configuration (#617-#620)

Create foundation with project metadata and basic structure.

### Files to Create

- `magic.toml` at repository root

### Sections to Include

- `[project]` section with metadata
- Empty/placeholder `[dependencies]` section
- Empty/placeholder `[channels]` section
- Header comments explaining file purpose

### Phase 2: Dependencies (#617-#620)

Add all required dependencies with proper versioning.

### Dependencies to Add

- MAX/Mojo packages with version constraints
- Python interpreter requirement
- Essential development packages

### Documentation Requirements

- Comment for each dependency explaining purpose
- Version constraint rationale
- Links to upstream documentation if needed

### Phase 3: Channels (#617-#620)

Configure package sources and channel priority.

### Channels to Configure

- conda-forge for standard packages
- Modular channel(s) for MAX/Mojo
- Channel priority specification

### Documentation Requirements

- Explain channel purpose
- Document priority rationale
- Note any channel-specific quirks

## API Contracts and Interfaces

### magic.toml Schema

The magic.toml file must conform to the Magic package manager expected schema:

```toml
# Project metadata section
[project]
name = "string"              # Required: package name
version = "string"           # Required: semantic version
description = "string"       # Required: project description
authors = ["string"]         # Optional: author list
license = "string"           # Optional: license identifier

# Dependencies section
[dependencies]
package-name = "version-spec"  # Format: ">=1.0", "~=1.5", "==1.2.3"

# Channels section
[channels]
channels = ["channel-name"]    # List of package sources
```text

### Integration Points

### Input Requirements

- Repository root directory must exist
- No conflicting magic.toml file present
- Write permissions on repository root

### Output Guarantees

- Valid TOML syntax
- Conformance to Magic schema
- Parseable by Magic package manager
- Human-readable and maintainable

### Dependencies

- Requires completion of directory structure creation
- Required by all downstream configuration that references packages
- Required by environment setup scripts

## Testing Strategy

Testing will be handled by issue #617. Key test scenarios:

1. **Syntax Validation**: TOML parser can read file without errors
1. **Schema Validation**: Magic can parse and validate configuration
1. **Dependency Resolution**: All specified packages are resolvable
1. **Environment Creation**: Can create environment from magic.toml
1. **Reproducibility**: Same configuration creates identical environment

## References

### Source Plans

- Parent Plan: [/notes/plan/01-foundation/02-configuration-files/plan.md](../../../../../../../notes/plan/01-foundation/02-configuration-files/plan.md)
- Component Plan: [/notes/plan/01-foundation/02-configuration-files/01-magic-toml/plan.md](../../../../../../../notes/plan/01-foundation/02-configuration-files/01-magic-toml/plan.md)
- Subcomponent Plans:
  - [Base Config](../../../../../../../notes/plan/01-foundation/02-configuration-files/01-magic-toml/01-create-base-config/plan.md)
  - [Dependencies](../../../../../../../notes/plan/01-foundation/02-configuration-files/01-magic-toml/02-add-dependencies/plan.md)
  - [Channels](../../../../../../../notes/plan/01-foundation/02-configuration-files/01-magic-toml/03-configure-channels/plan.md)

### Related Issues

- Issue #617: [Test] Magic TOML - Testing and Validation
- Issue #618: [Implementation] Magic TOML - Implementation
- Issue #619: [Package] Magic TOML - Integration and Packaging
- Issue #620: [Cleanup] Magic TOML - Cleanup and Finalization

### External Documentation

- [Magic Package Manager Documentation](https://docs.modular.com/magic/)
- [TOML Specification](https://toml.io/)
- [Conda Package Management](https://docs.conda.io/)
- [Semantic Versioning](https://semver.org/)

### Architectural Reviews

- See [/notes/review/](../../../../../../../notes/review/) for comprehensive architectural documentation
- See [/agents/](../../../../../../../agents/) for agent hierarchy and delegation patterns

## Implementation Notes

This section will be populated during the Test, Implementation, and Packaging phases with:

- Discoveries made during implementation
- Issues encountered and resolutions
- Deviations from original design (with rationale)
- Lessons learned for future components
