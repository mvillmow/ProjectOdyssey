# ml-odyssey

![Pre-commit Checks](https://github.com/mvillmow/ml-odyssey/actions/workflows/pre-commit.yml/badge.svg)
![Check Markdown Links](https://github.com/mvillmow/ml-odyssey/actions/workflows/link-check.yml/badge.svg)
![Agent Tests](https://github.com/mvillmow/ml-odyssey/actions/workflows/test-agents.yml/badge.svg)

Implementation of older AI papers for the modern age.

## Repository Structure

```text
ml-odyssey/
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

## Documentation

- [notes/README.md](notes/README.md) - Plan for creating GitHub issues
- [scripts/README.md](scripts/README.md) - Automation scripts documentation
- [agents/README.md](agents/README.md) - Agent system documentation
- [notes/issues/](notes/issues/) - Historical issue documentation (tracked)
- [notes/review/](notes/review/) - Review documentation (tracked)

**Note**: `notes/plan/` contains local planning files (not tracked in git). Reference tracked documentation
above for team collaboration.
