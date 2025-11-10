# ml-odyssey

![Pre-commit Checks](https://github.com/mvillmow/ml-odyssey/actions/workflows/pre-commit.yml/badge.svg)
![Check Markdown Links](https://github.com/mvillmow/ml-odyssey/actions/workflows/link-check.yml/badge.svg)
![Agent Tests](https://github.com/mvillmow/ml-odyssey/actions/workflows/test-agents.yml/badge.svg)
![Unit Tests](https://github.com/mvillmow/ml-odyssey/actions/workflows/unit-tests.yml/badge.svg)
![Integration Tests](https://github.com/mvillmow/ml-odyssey/actions/workflows/integration-tests.yml/badge.svg)
![Build Validation](https://github.com/mvillmow/ml-odyssey/actions/workflows/build-validation.yml/badge.svg)
![Security Scan](https://github.com/mvillmow/ml-odyssey/actions/workflows/security-scan.yml/badge.svg)

Implementation of older AI papers for the modern age.

## Repository Structure

```text
ml-odyssey/
├── papers/              # Research paper implementations
├── notes/
│   ├── plan/            # 4-level planning (LOCAL ONLY - not in git)
│   ├── issues/          # Issue documentation (tracked in git)
│   └── review/          # Review documentation (tracked in git)
├── agents/              # Agent documentation (tracked in git)
├── scripts/             # Automation scripts
│   ├── create_issues.py
│   └── README.md
└── logs/                # Script execution logs (not tracked)
```

## Quick Start

### Creating GitHub Issues

**Note**: Plan files are stored locally in `notes/plan/` and are NOT tracked in version control. They are
task-relative and used for local planning and GitHub issue generation.

To create GitHub issues from your local plan files:

```bash
# Test with dry-run first
python3 scripts/create_issues.py --dry-run

# Create all issues
python3 scripts/create_issues.py
```

See [scripts/README.md](scripts/README.md) for detailed documentation.

## Papers Directory

The `papers/` directory at the repository root contains all research paper implementations in Mojo. Each paper has its
own subdirectory with a standardized structure.

**Purpose**: Implementation of classic AI/ML research papers using Mojo for high performance.

**Structure**: Each paper implementation follows this organization:

```text
papers/
├── README.md          # Overview and guidelines
├── lenet-5/           # Example: LeNet-5 (1998)
│   ├── README.md      # Paper details and implementation notes
│   ├── src/           # Mojo implementation
│   ├── tests/         # Test suite
│   └── examples/      # Usage demonstrations
└── alexnet/           # Example: AlexNet (2012)
    ├── README.md
    ├── src/
    ├── tests/
    └── examples/
```

**Getting Started**:

The papers/ directory structure is already in place. To add a new paper implementation:

1. Create a subdirectory named after the paper (e.g., `papers/lenet5/`)
2. Follow the standard structure (src/, tests/, examples/, README.md)
3. See `papers/README.md` for detailed guidelines and conventions

**Packaging for Distribution**:

Create a distributable tarball of the papers/ directory:

```bash
python3 scripts/package_papers.py

# Or specify custom output directory
python3 scripts/package_papers.py --output /path/to/output
```

This creates `dist/papers-YYYYMMDD.tar.gz` containing the entire papers directory structure.

**Key Features**:

- Each paper is self-contained in its own directory
- Comprehensive README.md with paper overview and implementation notes
- Mojo-based implementations for maximum performance
- Complete test coverage and usage examples
- Distributable tarball packaging for easy sharing

## Language Selection Strategy

ML Odyssey uses a **Pragmatic Hybrid Approach** for language selection, documented in [ADR-001](notes/review/adr/ADR-001-language-selection-tooling.md).

### Mojo (Required)

**All ML/AI implementations use Mojo** for maximum performance:

- ✅ Neural network implementations (layers, activations, loss functions)
- ✅ Training loops and optimization algorithms
- ✅ Tensor operations and SIMD kernels
- ✅ Performance-critical data pipelines
- ✅ Model inference engines

**Why Mojo for ML/AI**: 10-100x faster than Python, type safety, memory safety, SIMD optimization.

### Python (Allowed)

**Automation scripts use Python** when Mojo limitations require it:

- ✅ GitHub API interaction (`gh` CLI requires subprocess output capture)
- ✅ Regex-heavy text processing (Mojo v0.25.7 has no regex support)
- ✅ CI/CD automation requiring process output

**Current Python scripts**:

- `scripts/create_issues.py` - GitHub issue automation (requires subprocess capture)
- `scripts/regenerate_github_issues.py` - Markdown parsing (requires regex)

**Conversion Plan**: Python scripts will be converted to Mojo when subprocess output capture and regex support
are available (target: Q2-Q3 2026).

See [ADR-001](notes/review/adr/ADR-001-language-selection-tooling.md) for complete strategy and rationale.

## Documentation

- [notes/README.md](notes/README.md) - Plan for creating GitHub issues
- [scripts/README.md](scripts/README.md) - Automation scripts documentation
- [agents/README.md](agents/README.md) - Agent system documentation
- [notes/issues/](notes/issues/) - Historical issue documentation (tracked)
- [notes/review/](notes/review/) - Review documentation (tracked)

**Note**: `notes/plan/` contains local planning files (not tracked in git). Reference tracked documentation
above for team collaboration.
