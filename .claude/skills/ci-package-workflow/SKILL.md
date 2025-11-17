---
name: ci-package-workflow
description: Create GitHub Actions workflows for automated package building and distribution. Use in package phase to automate .mojopkg building and release creation.
---

# CI Package Workflow Skill

Create CI workflows for automated packaging.

## When to Use

- Package phase of development
- Automating release process
- Building distributable packages
- Creating GitHub releases

## Workflow Structure

```yaml
name: Build Packages

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Mojo
        run: |
          # Install Mojo

      - name: Build Packages
        run: ./scripts/build_all_packages.sh

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          files: packages/*.mojopkg
```

## Common Workflows

### 1. Build on Tag

Trigger on version tags:

```yaml
on:
  push:
    tags:
      - 'v*.*.*'
```

### 2. Build on PR

Validate packaging on PR:

```yaml
on:
  pull_request:
    paths:
      - 'src/**'
      - 'scripts/build_*.sh'
```

### 3. Manual Trigger

Allow manual workflow runs:

```yaml
on:
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to build'
        required: true
```

## Best Practices

- Cache dependencies
- Upload artifacts
- Create GitHub releases
- Tag with version
- Test installation

See `.github/workflows/` for examples.
