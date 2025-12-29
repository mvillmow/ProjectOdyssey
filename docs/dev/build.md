# Consolidated Build & Installation

From root MDs (backed up `notes/root-backup/`). Links to [phases.md](phases.md).

## Package Building

**File**: `BUILD_PACKAGE.md`

**Training Module**:

- `mojo package shared/training -o dist/training-0.1.0.mojopkg`
- Verify: `./scripts/install_verify_training.sh`
- Automated: `./scripts/build_training_package.sh`

**Other Packages**: data/utils via scripts/build_*_package.sh.

## Installation

**File**: `INSTALL.md`

**Docker (Recommended)**:

```bash
git clone https://github.com/mvillmow/ProjectOdyssey.git
cd ProjectOdyssey
docker-compose up -d ProjectOdyssey-dev
docker-compose exec ProjectOdyssey-dev bash
pixi run pytest tests/
```

**Pixi Local**:

```bash
pixi install
pixi shell
pre-commit install
pytest tests/
```

**Workflow**: pre-commit, pytest, scripts/create_issues.py.

## Docker Details

**File**: `DOCKER.md` (summary): docker-compose.yml services (dev/ci/prod); volumes, ports.

## Build Instructions

**File**: `BUILD_INSTRUCTIONS.md`, `EXECUTE_BUILD.md`: pixi.toml deps, mojo package steps.

**Troubleshooting**: Docker perms, pixi cache rm, mojo version 0.26.1+.

Updated: 2025-11-24
