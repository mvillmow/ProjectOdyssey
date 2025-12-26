# Repository Structure Guide - Team Onboarding

<!-- markdownlint-disable MD024 -->

## Welcome to ML Odyssey

This guide helps you quickly navigate the ML Odyssey repository and understand where to find what you need.

## Quick Start: Finding Your Way

### "I'm New - Where Do I Start?"

### First Steps

1. **Read**: README.md (in repository root) - Project overview
1. **Install**: [installation.md](installation.md) - Set up environment
1. **Explore**: STRUCTURE.md (in repository root) - Repository organization
1. **Try**: examples/ directory (in repository root) - Working examples

### "I Want to Implement a Paper"

### Workflow

```bash
```bash

# 1. Generate structure

python tools/paper-scaffold/scaffold.py \
    --paper {name} \
    --title "{Title}" \
    --authors "{Authors}" \
    --year {year}

# 2. Configure experiment

cp configs/templates/experiment.yaml configs/experiments/{name}/baseline.yaml

# 3. Implement

cd papers/{name}/

# Edit model.mojo, train.mojo

# 4. Test

mojo test tests/papers/{name}/

# 5. Benchmark

mojo benchmarks/scripts/run_benchmarks.mojo --paper {name}

```text

**See**: `tools/paper-scaffold/README.md` for detailed instructions

### "I Want to Find Documentation"

**Documentation Locations**:

| Type | Location | Purpose |
| ------ | ---------- | --------- |
| **User Docs** | `docs/` | Tutorials, guides, API reference |
| **Issue Docs** | GitHub issue comments | Issue-specific implementation notes |
| **Architectural** | `notes/review/` | Design decisions, comprehensive specs |
| **ADRs** | `docs/adr/` | Architecture Decision Records |
| **Agent Docs** | `agents/` | Agent system documentation |
| **Code Comments** | Source files | Inline documentation |

**Quick Links**:

- API Reference → `docs/api/`
- Getting Started → `docs/getting-started/`
- Core Concepts → `docs/core/`
- Advanced Topics → `docs/advanced/`

### "I Want to Run Tests"

**Test Locations**:

```bash

```bash

# All tests
mojo test tests/

# Specific subsystem
mojo test tests/shared/          # Shared library tests
mojo test tests/papers/lenet5/   # Paper-specific tests
mojo test tests/foundation/      # Foundation tests

# With coverage
python tools/testing/coverage.py

```text

**See**: `tests/README.md` for testing guidelines

### "I Want to Measure Performance"

### Benchmarking

```bash
```bash

# Run all benchmarks

mojo benchmarks/scripts/run_benchmarks.mojo

# Run specific benchmark

mojo benchmarks/scripts/lenet5_benchmark.mojo

# Compare with baseline

mojo benchmarks/scripts/compare_results.mojo \
    --baseline benchmarks/baselines/baseline_results.json \
    --current benchmarks/results/latest_results.json

```text

**See**: `benchmarks/README.md` for details

## Repository Organization

### Top-Level Directories

```text

```text

ProjectOdyssey/
├── papers/          # ML paper implementations
├── shared/          # Reusable ML components
├── benchmarks/      # Performance benchmarking
├── docs/            # User documentation
├── agents/          # AI agent system docs
├── tools/           # Development utilities
├── configs/         # Configuration management
├── tests/           # Test suite
├── examples/        # Usage examples
├── scripts/         # Automation scripts
├── notes/           # Planning and architectural docs
├── .claude/         # Claude Code configurations
└── .github/         # CI/CD workflows

```text

### Core Directories Explained

#### papers/ - Research Implementations

**What**: ML research paper implementations

### Structure

- Each paper in its own directory
- `_template/` for starting new papers

### Example

```text
```text

papers/
├── _template/
├── lenet5/
│   ├── model.mojo
│   ├── train.mojo
│   ├── tests/
│   └── README.md

```text

**When to Use**: Implementing or studying paper implementations

#### shared/ - Reusable Components

**What**: Core ML components used across papers

**Structure**:

- `core/` - Layers, activations, loss functions
- `training/` - Optimizers, schedulers
- `data/` - Data loaders
- `utils/` - Utilities

**When to Use**: Building models, training, loading data

#### Supporting Directories (The Big 5)

##### 1. benchmarks/ - Performance Measurement

**Purpose**: Track and compare performance

**Key Files**:

- `scripts/` - Benchmark execution
- `baselines/` - Baseline results
- `results/` - Timestamped results

**Common Tasks**:

```bash

```bash

# Run benchmarks
mojo benchmarks/scripts/run_benchmarks.mojo

# Compare results
mojo benchmarks/scripts/compare_results.mojo

```text

##### 2. docs/ - User Documentation

**Purpose**: Comprehensive documentation for all users

### Key Sections

- `getting-started/` - Onboarding
- `core/` - Core concepts
- `advanced/` - Advanced topics
- `dev/` - Developer docs

### Common Tasks

- Read guides: Browse `docs/`
- Add docs: Create in appropriate section
- Validate: `python scripts/validate_links.py docs/`

##### 3. agents/ - AI Agent System

**Purpose**: Agent hierarchy documentation

### Key Files

- `hierarchy.md` - Agent levels
- `delegation-rules.md` - Coordination
- `templates/` - Agent templates

### Common Tasks

- Understand agents: Read `hierarchy.md`
- Create agent: Use `templates/`
- Find agent configs: Check `.claude/agents/`

##### 4. tools/ - Development Utilities

**Purpose**: Developer productivity tools

### Key Tools

- `paper-scaffold/` - Generate paper structure
- `test-utils/` - Testing utilities
- `benchmarking/` - Benchmark framework
- `codegen/` - Code generation

### Common Tasks

```bash
```bash

# Scaffold paper

python tools/paper-scaffold/scaffold.py --paper {name}

# Generate boilerplate

python tools/codegen/mojo_boilerplate.py --layer Conv2D

```text

##### 5. configs/ - Configuration Management

**Purpose**: Centralized experiment configuration

**Key Sections**:

- `defaults/` - Default settings
- `papers/` - Paper-specific configs
- `experiments/` - Experiment variations
- `templates/` - Config templates

**Common Tasks**:

```bash

```bash

# Create experiment
cp configs/templates/experiment.yaml configs/experiments/{paper}/{name}.yaml

# Validate config
python scripts/lint_configs.py configs/experiments/{paper}/{name}.yaml

```text

## Common Workflows

### Workflow 1: I Want to Start a New Paper

### Steps

1. **Scaffold**:

   ```bash
   python tools/paper-scaffold/scaffold.py \
       --paper resnet \
       --title "Deep Residual Learning" \
       --authors "He et al." \
       --year 2015
   ```text

1. **Configure**:

   ```bash
   # Copy and edit configs
   cp configs/templates/paper.yaml configs/papers/resnet/model.yaml
   cp configs/templates/experiment.yaml configs/experiments/resnet/baseline.yaml
   ```text

1. **Implement**:

   ```bash
   cd papers/resnet/
   # Edit model.mojo, train.mojo
   ```text

1. **Test**:

   ```bash
   mojo test tests/papers/resnet/
   ```text

1. **Document**:

   ```bash
   # Update papers/resnet/README.md
   # Add docs/api/papers/resnet.md
   ```text

### Workflow 2: I Want to Add a Reusable Component

### Steps

1. **Implement**:

   ```bash
   # Add to appropriate location
   vim shared/core/layers/attention.mojo
   ```text

1. **Test**:

   ```bash
   # Add tests
   vim tests/shared/core/layers/test_attention.mojo
   mojo test tests/shared/core/layers/
   ```text

1. **Document**:

   ```bash
   # Update API docs
   vim docs/api/shared/layers.md
   ```text

1. **Example**:

   ```bash
   # Add usage example
   vim examples/custom_layer/attention_example.mojo
   ```text

### Workflow 3: I Want to Run an Experiment

### Steps

1. **Create Config**:

   ```bash
   cp configs/experiments/lenet5/baseline.yaml \
      configs/experiments/lenet5/augmented.yaml
   # Edit augmented.yaml with changes
   ```text

1. **Run Training**:

   ```bash
   mojo papers/lenet5/train.mojo --config experiments/lenet5/augmented
   ```text

1. **Benchmark**:

   ```bash
   mojo benchmarks/scripts/run_benchmarks.mojo \
       --experiment lenet5/augmented
   ```text

1. **Document**:

   ```bash
   # Record results
   vim docs/research/experiments/lenet5_augmentation.md
   ```text

### Workflow 4: I Want to Optimize Performance

### Steps

1. **Measure**:

   ```bash
   mojo benchmarks/scripts/run_benchmarks.mojo --paper lenet5
   ```text

1. **Profile**:

   ```bash
   mojo tools/benchmarking/runner.mojo --target lenet5 --profile
   ```text

1. **Optimize**:

   ```bash
   # Edit code based on profiling
   vim papers/lenet5/model.mojo
   ```text

1. **Verify**:

   ```bash
   mojo benchmarks/scripts/run_benchmarks.mojo --paper lenet5 --compare
   ```text

1. **Document**:

   ```bash
   vim docs/advanced/optimization_techniques.md
   ```text

## Decision Tree: Where Does This Go

### Content Type Decision Tree

### Q: What type of content are you adding?

```text
```text

ML Paper Implementation?
├─ Yes → papers/{paper_name}/
└─ No → Continue

Reusable ML Component?
├─ Yes → shared/{core|training|data|utils}/
└─ No → Continue

User Documentation?
├─ Yes → docs/{getting-started|core|advanced|dev}/
└─ No → Continue

Performance Benchmark?
├─ Yes → benchmarks/scripts/
└─ No → Continue

Development Tool?
├─ Yes → tools/{category}/
└─ No → Continue

Configuration File?
├─ Yes → configs/{defaults|papers|experiments}/
└─ No → Continue

Test File?
├─ Yes → tests/{foundation|shared|papers|tools}/
└─ No → Continue

Usage Example?
├─ Yes → examples/
└─ No → Continue

Automation Script?
├─ Yes → scripts/
└─ No → Continue

Agent Documentation?
├─ Yes → agents/ (docs) or .claude/agents/ (configs)
└─ No → Continue

Issue-Specific Notes?
├─ Yes → Post as GitHub issue comment (gh issue comment <number>)
└─ No → Continue

Architectural Decision?
├─ Yes → docs/adr/ (for ADRs) or notes/review/ (for specs)
└─ No → Ask in team channel!

```text

## Best Practices

### When Adding New Content

1. **Check Existing Structure** - Don't duplicate
2. **Follow Conventions** - Use established patterns
3. **Add READMEs** - Every directory needs documentation
4. **Update Indexes** - Link from appropriate index files
5. **Run Validation** - Check structure and links

### When Writing Code

1. **Use Mojo for ML** - Performance-critical code in Mojo
2. **Use Python for Automation** - Tools and scripts (with justification)
3. **Follow TDD** - Write tests first
4. **Document As You Go** - Don't defer documentation
5. **Benchmark Changes** - Verify performance impact

### When Documenting

1. **Choose Right Location**:
   - User-facing → `docs/`
   - Implementation notes → GitHub issue comments
   - Architecture decisions → `docs/adr/`
   - Comprehensive specs → `notes/review/`

2. **Follow Markdown Standards**:
   - Specify language for code blocks
   - Blank lines around blocks and lists
   - Max 120 character lines

3. **Link Related Content**:
   - Cross-reference related docs
   - Link to source code
   - Reference issues and PRs

## Troubleshooting

### "I Can't Find Where to Put My Code"

**Solution**: Use the decision tree above or ask in team channel

**Quick Check**:

- ML implementation? → `papers/` or `shared/`
- Tool/utility? → `tools/`
- Configuration? → `configs/`
- Test? → `tests/`

### "I Don't Know Which Config to Use"

**Solution**: Use the 3-level hierarchy

**Hierarchy**:

1. `configs/defaults/` - Base settings
2. `configs/papers/{paper}/` - Paper overrides
3. `configs/experiments/{paper}/{exp}/` - Experiment overrides

**Example**:

```bash

```bash

# For baseline reproduction
mojo papers/lenet5/train.mojo --config experiments/lenet5/baseline

# For custom experiment
mojo papers/lenet5/train.mojo --config experiments/lenet5/my_experiment

```text

### "My Links Are Broken"

**Solution**: Run link validation

```bash
```bash

# Check all docs

python scripts/validate_links.py docs/

# Check specific file

python scripts/validate_links.py docs/path/to/file.md

```text

**Common Issues**:

- Wrong relative path
- File moved/renamed
- Missing file extension

### "Structure Validation Fails"

**Solution**: Run structure validator

```bash

```bash

# Check entire repository
python scripts/validate_structure.py

# Check what's missing
python scripts/validate_structure.py --verbose

```text

## Quick Reference

### Essential Commands

```bash
```bash

# Setup environment

python scripts/setup.py

# Validate structure

python scripts/validate_structure.py

# Validate documentation

python scripts/check_readmes.py
python scripts/validate_links.py

# Run tests

mojo test tests/

# Run benchmarks

mojo benchmarks/scripts/run_benchmarks.mojo

# Format code

make pre-commit-all

```text

### Essential Files

| File | Purpose |
| ------ | --------- |
| `README.md` | Project overview |
| `STRUCTURE.md` | Repository organization |
| `CONTRIBUTING.md` | Contribution guidelines |
| `CLAUDE.md` | Claude Code conventions |
| `docs/index.md` | Documentation index |

### Essential Directories

| Directory | Quick Description |
| ----------- | ------------------- |
| `papers/` | ML implementations |
| `shared/` | Reusable components |
| `benchmarks/` | Performance tracking |
| `docs/` | User documentation |
| `tools/` | Developer utilities |
| `configs/` | Experiment configs |
| `tests/` | Test suite |

## Next Steps

### For New Contributors

1. **Setup**: Follow installation guide
2. **Explore**: Browse examples/ directory in repository root
3. **Read**: Check `docs/core/` for concepts
4. **Try**: Implement a simple example
5. **Ask**: Use team channels for questions

### For Implementers

1. **Scaffold**: Use `tools/paper-scaffold/`
2. **Configure**: Set up `configs/`
3. **Implement**: Write in `papers/`
4. **Test**: Add to `tests/`
5. **Benchmark**: Measure in `benchmarks/`

### For Documentation Writers

1. **Identify**: Find documentation gaps
2. **Write**: Create in appropriate `docs/` section
3. **Link**: Update `docs/index.md`
4. **Validate**: Run `scripts/validate_links.py`
5. **Review**: Submit PR for team review

## Getting Help

### Documentation

- **This Guide**: Repository navigation
- **STRUCTURE.md**: Complete directory reference
- **docs/**: Comprehensive documentation
- **agents/**: Agent system documentation

### Team Resources

- Team channels for questions
- GitHub issues for bugs/features
- Pull request reviews for feedback
- Weekly team meetings for discussions

## Summary

**Key Takeaways**:

1. **Organization**: Repository is logically organized by purpose
2. **Locations**: Use decision tree to find right location
3. **Tools**: Use `tools/` to automate repetitive tasks
4. **Configs**: Use 3-level hierarchy for experiments
5. **Validation**: Run validation scripts before committing

**Remember**:

- `papers/` - Implementations
- `shared/` - Reusable components
- `benchmarks/` - Performance
- `docs/` - Documentation
- `tools/` - Utilities
- `configs/` - Configuration

**When in Doubt**: Check STRUCTURE.md in repository root or ask the team!

---

**Last Updated**: 2025-11-16
**Maintained By**: Documentation Specialist
