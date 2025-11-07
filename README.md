# ml-odyssey
Implementation of older AI papers for the modern age.

## Repository Structure

```
ml-odyssey/
├── notes/plan/          # 4-level hierarchical planning structure
│   ├── 01-foundation/
│   ├── 02-shared-library/
│   ├── 03-tooling/
│   ├── 04-first-paper/
│   ├── 05-ci-cd/
│   └── 06-agentic-workflows/
├── scripts/             # Automation scripts
│   ├── create_issues.py
│   └── README.md
└── logs/                # Script execution logs
```

## Quick Start

### Creating GitHub Issues

All planning has been documented in `notes/plan/` with 331 components across 6 major sections. To create GitHub issues:

```bash
# Test with dry-run first
python3 scripts/create_issues.py --dry-run

# Create all 1,655 issues
python3 scripts/create_issues.py
```

See [scripts/README.md](scripts/README.md) for detailed documentation.

## Documentation

- [notes/README.md](notes/README.md) - Plan for creating GitHub issues
- [scripts/README.md](scripts/README.md) - Automation scripts documentation
- [scripts/SCRIPTS_ANALYSIS.md](scripts/SCRIPTS_ANALYSIS.md) - Comprehensive scripts analysis
- [notes/plan/](notes/plan/) - Complete hierarchical planning structure
