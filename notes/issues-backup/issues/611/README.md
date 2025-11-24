# Issue #611: [Plan] Configure Channels - Design and Documentation

## Objective

Configure package channels in magic.toml to specify where Magic should look for packages, including conda-forge and Modular-specific channels needed for Mojo/MAX packages.

## Deliverables

- magic.toml with channels section
- Package sources configured correctly
- Channel priority set appropriately

## Success Criteria

- [ ] Channels section is properly configured
- [ ] All necessary package sources are included
- [ ] Channel priority is set correctly
- [ ] Configuration allows finding all dependencies

## Design Decisions

### Channel Selection

**Decision**: Use conda-forge as primary channel with Modular-specific channels for Mojo/MAX packages.

### Rationale

- conda-forge is the standard, community-maintained package repository with extensive package availability
- Modular channels are required for Mojo/MAX-specific packages not available in conda-forge
- This combination ensures both general Python/ML packages and Mojo-specific tools are accessible

### Alternatives Considered

- Using only conda-forge: Would miss Mojo/MAX-specific packages
- Using only Modular channels: Would require duplicating all general packages
- Using defaults channel: conda-forge is more current and comprehensive

### Channel Priority Strategy

**Decision**: Set explicit channel priority to ensure deterministic package resolution.

### Rationale

- Prevents conflicts when packages exist in multiple channels
- Ensures consistent builds across different environments
- Allows Modular channels to override general packages when needed

### Configuration Approach

- Primary: conda-forge (for general packages)
- Secondary: Modular channels (for Mojo/MAX packages)
- Priority can be adjusted based on package availability and version requirements

### Configuration Structure

**Decision**: Use Magic's TOML-based channel configuration format.

### Format

```toml
[channels]
# List channels in priority order (highest to lowest)
channels = [
    "conda-forge",
    # Add Modular channels as needed for Mojo/MAX
]
```text

### Rationale

- Follows Magic's native configuration format
- Simple, readable, and maintainable
- Allows easy updates as new channels become available

## Architecture

### Component Overview

This configuration is part of the foundation layer, establishing the package management infrastructure for the entire project.

### Dependencies

### Inputs

- Existing magic.toml file with project metadata
- Existing dependencies section (from issue #610)
- Knowledge of required package sources

### Outputs

- Configured channels section in magic.toml
- Validated package resolution capability

### Integration Points

- Works with dependencies configuration to resolve packages
- Enables subsequent installation and environment setup
- Foundation for all package management operations

## Implementation Strategy

### Step-by-Step Approach

1. **Add channels section** to magic.toml
   - Insert after project metadata and before/after dependencies
   - Use array format for channel list

1. **Configure conda-forge channel**
   - Primary channel for Python, NumPy, testing frameworks, etc.
   - Ensures access to standard ML/AI packages

1. **Add Modular channels**
   - Required for Mojo/MAX-specific packages
   - Research exact channel names from Modular documentation

1. **Set channel priority**
   - Verify priority order resolves all dependencies
   - Test that conflicts are resolved correctly

### Validation

- Verify magic.toml syntax is valid
- Test package resolution with configured channels
- Ensure all dependencies can be found
- Confirm no channel conflicts

## References

- Source Plan: [notes/plan/01-foundation/02-configuration-files/01-magic-toml/03-configure-channels/plan.md](notes/plan/01-foundation/02-configuration-files/01-magic-toml/03-configure-channels/plan.md)
- Parent Component: Magic.toml Configuration
- Related Issues:
  - #612: [Test] Configure Channels
  - #613: [Implementation] Configure Channels
  - #614: [Packaging] Configure Channels
  - #615: [Cleanup] Configure Channels
  - #610: Add Dependencies (prerequisite)

## Implementation Notes

(To be filled during implementation phase)
