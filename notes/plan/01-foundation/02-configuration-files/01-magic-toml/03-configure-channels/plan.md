# Configure Channels

## Overview

Configure package channels in magic.toml to specify where Magic should look for packages. This includes setting up conda-forge and any Modular-specific channels needed for Mojo/MAX packages.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- magic.toml with dependencies exists
- Knowledge of available package channels
- Understanding of channel priority

## Outputs

- magic.toml with channels section
- Package sources configured correctly
- Channel priority set appropriately

## Steps

1. Add channels section to magic.toml
2. Configure conda-forge channel for general packages
3. Add Modular channels for Mojo/MAX packages
4. Set channel priority if needed

## Success Criteria

- [ ] Channels section is properly configured
- [ ] All necessary package sources are included
- [ ] Channel priority is set correctly
- [ ] Configuration allows finding all dependencies

## Notes

Channels determine where packages are downloaded from. Use standard channels like conda-forge plus any Modular-specific channels needed for MAX.
