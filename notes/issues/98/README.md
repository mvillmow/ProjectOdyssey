# Issue #98: [Plan] Configure Channels - Design and Documentation

## Objective

Configure package channels in magic.toml to specify where Magic should look for packages, including conda-forge and Modular-specific channels for Mojo/MAX packages with appropriate channel priority.

## Deliverables

- Comprehensive design specification for channels configuration
- Channel selection strategy and priority ordering
- Documentation of available Modular channels and their purposes
- Implementation guidelines for adding channels to magic.toml
- Validation requirements for channel configuration

## Success Criteria

- [x] Channels section structure fully specified
- [x] Channel priority strategy documented
- [x] All necessary package sources identified
- [x] Configuration enables finding all dependencies
- [x] Clear guidelines for future channel additions
- [x] Validation requirements defined

## References

- [Foundation Plan](../../plan/01-foundation/plan.md) - Repository foundation overview
- [Magic TOML Plan](../../plan/01-foundation/02-configuration-files/01-magic-toml/plan.md) - Magic configuration strategy
- [Configure Channels Plan](../../plan/01-foundation/02-configuration-files/01-magic-toml/03-configure-channels/plan.md) - Component requirements
- [Issue #87 Base Config](../87/README.md) - Base magic.toml structure
- [CLAUDE.md](../../../CLAUDE.md) - Project conventions and guidelines

## Planning Documentation

### 1. Channels Configuration Overview

**Purpose**: Channels in `magic.toml` define the sources from which Magic downloads packages. They are similar to npm registries or PyPI mirrors, but follow the conda ecosystem model.

**Key Concepts**:
- **Channel**: A repository of packages (conda packages, Mojo packages, etc.)
- **Priority**: Order in which channels are searched for packages
- **URL Format**: Can be channel names (e.g., "conda-forge") or full URLs
- **Resolution**: Magic searches channels in priority order until package is found

### 2. Channel Selection Strategy

#### 2.1 Required Channels

**Modular Channels** (for Mojo/MAX packages):
- **Stable Channel**: `https://conda.modular.com/max`
  - Contains stable releases of MAX and Mojo
  - Recommended for production use
  - Updated with each official release
  
- **Nightly Channel**: `https://conda.modular.com/max-nightly`
  - Contains daily builds with latest features
  - Useful for testing new capabilities
  - May have breaking changes
  - Currently used in pixi.toml

**Community Channels** (for general packages):
- **conda-forge**: `conda-forge`
  - Largest community-driven conda channel
  - Contains Python, NumPy, matplotlib, and most scientific packages
  - Well-maintained and regularly updated
  - De facto standard for conda packages

#### 2.2 Optional Channels (for future consideration)

- **defaults**: Anaconda's default channel (proprietary, licensing considerations)
- **pytorch**: For PyTorch packages if needed
- **nvidia**: For CUDA-specific packages
- Custom internal channels for proprietary packages

### 3. Channel Configuration Design

#### 3.1 Recommended Configuration

```toml
[channels]
# Channel priority is determined by order (first = highest priority)
# Magic searches channels in order until package is found
default = [
    "https://conda.modular.com/max",      # Modular stable channel for MAX/Mojo
    "conda-forge",                        # Community packages
]

# Alternative configuration for development/testing
# default = [
#     "https://conda.modular.com/max-nightly",  # Latest Mojo/MAX features
#     "conda-forge",
# ]
```

#### 3.2 Configuration Variations

**For Stable Development** (Recommended):
```toml
[channels]
default = [
    "https://conda.modular.com/max",
    "conda-forge",
]
```

**For Cutting-Edge Features**:
```toml
[channels]
default = [
    "https://conda.modular.com/max-nightly",
    "conda-forge",
]
```

**With Additional Channels**:
```toml
[channels]
default = [
    "https://conda.modular.com/max",
    "conda-forge",
    "pytorch",  # If PyTorch needed for comparisons
]
```

### 4. Channel Priority and Resolution

#### 4.1 Priority Order

Channels are searched in the order listed:
1. First channel in list = highest priority
2. Subsequent channels searched only if package not found
3. First match wins (no version comparison across channels)

#### 4.2 Resolution Strategy

When Magic looks for a package:
1. Search first channel for exact package name
2. If found, use that version (even if newer exists elsewhere)
3. If not found, search next channel
4. Continue until package found or channels exhausted
5. Error if package not found in any channel

#### 4.3 Best Practices

- **Modular channel first**: Ensures official Mojo/MAX packages are used
- **Minimize channels**: Fewer channels = faster resolution
- **Avoid channel conflicts**: Don't mix incompatible channel sets
- **Document channel choices**: Explain why each channel is included

### 5. Implementation Guidelines

#### 5.1 Adding to Existing magic.toml

The channels section should be added after the project metadata and before dependencies:

```toml
# ML Odyssey - Magic Package Manager Configuration
# ... existing header comments ...

[project]
# ... existing project metadata ...

[project.urls]
# ... existing URLs ...

# Package channels configuration
# Channels are searched in order for packages (first match wins)
[channels]
default = [
    "https://conda.modular.com/max",      # Official Modular channel for MAX/Mojo packages
    "conda-forge",                        # Community-driven channel for general packages
]

# Note: Dependencies section follows
# [dependencies]
# ... (added in Issue #92) ...
```

#### 5.2 Channel Documentation

Each channel entry should include comments explaining:
- What packages it provides
- Why it's included
- Its priority reasoning

Example:
```toml
[channels]
default = [
    # Modular's official channel - provides MAX, Mojo compiler, and related tools
    # Must be first to ensure official packages are preferred
    "https://conda.modular.com/max",
    
    # Community channel - provides Python, NumPy, matplotlib, and scientific packages
    # Well-maintained with broad package coverage
    "conda-forge",
]
```

### 6. Validation Requirements

#### 6.1 Syntax Validation

- Valid TOML syntax for arrays
- Proper string quoting for URLs
- No trailing commas in arrays
- Correct section naming (`[channels]` not `[channel]`)

#### 6.2 Content Validation

**Channel URLs**:
- Must be valid URLs or recognized channel names
- HTTPS required for custom URLs (security)
- No duplicate channels in list
- At least one channel must be specified

**Channel Accessibility**:
- Channels should be reachable (network test)
- Valid conda channel structure
- Contains repodata.json (channel index)

#### 6.3 Compatibility Validation

- Modular channel must be compatible with MAX version in dependencies
- Channel set must provide all required packages
- No conflicting package versions across channels

### 7. Testing Strategy

#### 7.1 Unit Tests

```python
def test_channels_section_exists():
    """Verify channels section is present in magic.toml"""
    config = parse_toml("magic.toml")
    assert "channels" in config
    assert "default" in config["channels"]

def test_modular_channel_included():
    """Verify Modular channel is configured"""
    channels = config["channels"]["default"]
    assert any("conda.modular.com" in ch for ch in channels)

def test_conda_forge_included():
    """Verify conda-forge is configured"""
    channels = config["channels"]["default"]
    assert "conda-forge" in channels

def test_channel_priority():
    """Verify Modular channel has higher priority than conda-forge"""
    channels = config["channels"]["default"]
    modular_idx = next(i for i, ch in enumerate(channels) if "modular" in ch)
    forge_idx = channels.index("conda-forge")
    assert modular_idx < forge_idx
```

#### 7.2 Integration Tests

```bash
# Test channel resolution
magic search mojo  # Should find in Modular channel
magic search numpy  # Should find in conda-forge

# Test package installation
magic install mojo  # Should come from Modular channel
magic install python  # Should come from conda-forge
```

#### 7.3 Validation Tests

```bash
# Validate TOML syntax
python -c "import tomli; tomli.load(open('magic.toml', 'rb'))"

# Check channel accessibility
for channel in $(magic config show channels); do
    curl -I "$channel/repodata.json"
done
```

### 8. Migration Considerations

#### 8.1 From pixi.toml

Current pixi.toml uses:
```toml
channels = ["https://conda.modular.com/max-nightly", "conda-forge"]
```

Migration decision points:
1. **Stable vs Nightly**: Start with stable channel for magic.toml
2. **Channel order**: Maintain same priority (Modular first)
3. **Additional channels**: Add only as needed

#### 8.2 Channel Switching

To switch between stable and nightly:
1. Update channel URL in magic.toml
2. Clear package cache: `magic clean --all`
3. Re-resolve dependencies: `magic install`

### 9. Troubleshooting Guide

#### Common Issues and Solutions

**Issue**: Package not found
- **Check**: Is package name correct?
- **Check**: Is appropriate channel configured?
- **Solution**: Add required channel or verify package availability

**Issue**: Wrong package version installed
- **Check**: Channel priority order
- **Check**: Version constraints in dependencies
- **Solution**: Reorder channels or pin specific versions

**Issue**: Slow package resolution
- **Check**: Number of channels configured
- **Check**: Network connectivity to channels
- **Solution**: Remove unnecessary channels, use faster mirrors

**Issue**: Channel authentication errors
- **Check**: Channel URL correctness
- **Check**: Network proxy settings
- **Solution**: Verify URL, configure proxy if needed

### 10. Future Enhancements

#### 10.1 Named Channel Groups

Future magic.toml might support:
```toml
[channels.production]
default = ["https://conda.modular.com/max", "conda-forge"]

[channels.development]
default = ["https://conda.modular.com/max-nightly", "conda-forge"]
```

#### 10.2 Per-Package Channel Selection

```toml
[dependencies]
mojo = { version = ">=24.5", channel = "max-nightly" }
python = { version = "3.11", channel = "conda-forge" }
```

#### 10.3 Channel Mirrors

```toml
[channels]
default = ["max", "conda-forge"]

[channels.mirrors]
max = ["https://mirror1.example.com/max", "https://conda.modular.com/max"]
```

## Implementation Notes

**Phase**: Planning (Issue #98 - Current)
**Dependencies**: Issue #87 (Base Config) completed planning
**Next Phases**: Test, Implementation, Package, Cleanup (Issues TBD)

### Key Decisions

1. **Use stable channel by default**: More reliable for development
2. **Modular channel first**: Ensures official packages are preferred
3. **Only essential channels**: Start with minimal set, expand as needed
4. **Clear documentation**: Every channel choice explained

### Design Rationale

**Why stable over nightly?**
- Reproducible builds
- Fewer breaking changes
- Better for learning/research
- Can switch to nightly when needed

**Why conda-forge over defaults?**
- Open source, no licensing issues
- Larger package selection
- Community-driven
- Better maintained

**Why explicit URLs?**
- Clear about exact source
- No ambiguity in channel names
- Easier to switch between stable/nightly
- Self-documenting

### Success Validation

The planning phase is complete when:
- ✅ Channel configuration structure fully specified
- ✅ Channel priority strategy documented
- ✅ Available channels catalogued with purposes
- ✅ Implementation approach defined
- ✅ Testing strategy outlined
- ✅ Migration path from pixi.toml considered
- ✅ Troubleshooting guide provided

## Next Steps

1. **Test Phase**: Write tests for channel configuration validation
2. **Implementation Phase**: Add channels section to magic.toml
3. **Package Phase**: Integrate channel configuration into workflow
4. **Cleanup Phase**: Optimize and document final configuration

---

**Last Updated**: 2025-11-15
**Status**: Planning Complete ✅
