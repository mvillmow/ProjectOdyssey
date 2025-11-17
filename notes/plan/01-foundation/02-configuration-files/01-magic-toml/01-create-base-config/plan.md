# Create Base Config

## Overview

Create the basic magic.toml file with project metadata including name, version, description, and basic structure. This establishes the foundation for the Magic package manager configuration.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Repository root directory exists
- Project name and description defined
- Understanding of magic.toml structure

## Outputs

- magic.toml file at repository root
- Project section with metadata
- Basic file structure for adding dependencies and channels

## Steps

1. Create magic.toml file at repository root
2. Add project section with name, version, and description
3. Set up basic structure for dependencies and channels sections
4. Add comments explaining each section

## Success Criteria

- [ ] magic.toml file exists at repository root
- [ ] Project metadata is complete and accurate
- [ ] File structure is valid TOML
- [ ] Comments explain the configuration

## Notes

Start with a minimal valid configuration. The file will be expanded in subsequent steps with dependencies and channels.
