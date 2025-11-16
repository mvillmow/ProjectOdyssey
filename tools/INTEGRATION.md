# Tools Integration Guide

This guide explains how the `tools/` directory integrates with the ML Odyssey repository workflow.

## Overview

The tools directory provides development utilities that complement the repository's automation infrastructure:

- **`tools/`**: Interactive development utilities (this directory)
- **`scripts/`**: Repository automation and CI/CD scripts
- **`.claude/agents/`**: Agent configurations and specifications
- **`.github/workflows/`**: CI/CD pipeline definitions

## Integration Points

### 1. Development Workflow Integration

Tools integrate directly into the paper implementation workflow:

```text
Developer Workflow:
1. Create paper structure → tools/paper-scaffold/
2. Write tests → tools/test-utils/ (fixtures, generators)
3. Implement model → (direct Mojo implementation)
4. Benchmark performance → tools/benchmarking/
5. Generate boilerplate → tools/codegen/
```

### 2. Scripts Directory Integration

Clear separation of concerns between tools and scripts:

| Aspect | tools/ | scripts/ |
|--------|--------|----------|
| **Purpose** | Development utilities | Repository automation |
| **User** | Developers during implementation | CI/CD, maintainers |
| **Usage Pattern** | Interactive, on-demand | Automated, scheduled |
| **Examples** | scaffold.py, benchmark.mojo | create_issues.py, validate_configs.sh |

**Integration Pattern**:
- Scripts may invoke tools for validation (e.g., `scripts/validate_configs.sh` could use `tools/validation/`)
- Tools do NOT invoke scripts (separation of concerns)
- Both respect repository conventions (language selection, file structure)

### 3. CI/CD Pipeline Integration

Tools can be integrated into GitHub Actions workflows:

#### Example: Benchmarking in CI

```yaml
# .github/workflows/benchmark.yml
- name: Run performance benchmarks
  run: |
    mojo tools/benchmarking/model_bench.mojo
    python tools/benchmarking/report_generator.py
```

#### Example: Code Generation Validation

```yaml
# .github/workflows/codegen-validation.yml
- name: Validate generated code
  run: |
    python tools/codegen/mojo_boilerplate.py --validate
```

#### Current Integration Status

**Existing Workflows** (`.github/workflows/`):
- `pre-commit.yml` - Could integrate validation tools
- `test-agents.yml` - Could use test utilities
- `unit-tests.yml` - Could leverage test-utils fixtures
- `benchmark.yml` - Could use benchmarking tools
- `validate-configs.yml` - Could use validation utilities

### 4. Agent System Integration

Tools support agent-driven workflows (`.claude/agents/`):

**Agent Tool Usage**:
- **Planning Agents**: Reference tool capabilities in specifications
- **Implementation Agents**: Use tools for code generation and scaffolding
- **Test Agents**: Leverage test utilities and fixtures
- **Package Agents**: Use tools for distribution preparation

**Example Agent Integration**:
```markdown
## Skills to Use

- [`generate_boilerplate`](../skills/tier-1/generate-boilerplate/SKILL.md)
  → Wraps `tools/codegen/` for agent use
- [`run_tests`](../skills/tier-1/run-tests/SKILL.md)
  → Uses `tools/test-utils/` for test execution
```

## Usage Scenarios

### Scenario 1: Creating a New Paper Implementation

**Workflow**:
```bash
# 1. Scaffold paper structure
python tools/paper-scaffold/scaffold.py \
    --paper "AlexNet" \
    --author "Krizhevsky et al." \
    --year 2012 \
    --output papers/alexnet/

# 2. Generate model boilerplate
python tools/codegen/mojo_boilerplate.py \
    --type layer \
    --name Conv2D \
    --output papers/alexnet/model.mojo

# 3. Use test utilities
# (Import fixtures in test files)
from tools.test_utils import generate_batch, ModelFixture
```

### Scenario 2: Running Benchmarks

**Workflow**:
```bash
# 1. Benchmark model performance
mojo tools/benchmarking/model_bench.mojo \
    --paper lenet5 \
    --batch-sizes 1,8,32,128

# 2. Generate report
python tools/benchmarking/report_generator.py \
    --input benchmarks/lenet5.json \
    --output benchmarks/lenet5_report.html
```

### Scenario 3: Test-Driven Development

**Workflow**:
```mojo
// 1. Use test data generators
from tools.test_utils import generate_batch

fn test_forward_pass():
    let batch = generate_batch(shape=(32, 3, 28, 28))
    let model = MyModel()
    let output = model.forward(batch)
    assert output.shape == (32, 10)

// 2. Use performance utilities
from tools.test_utils import measure_latency

fn test_inference_speed():
    let model = MyModel()
    let latency = measure_latency(model, num_runs=100)
    assert latency < 10.0  # milliseconds
```

## Configuration and Setup

### Environment Detection

Tools respect repository configuration:

```bash
# Tools automatically detect:
- Repository root (via git)
- Mojo version (via `mojo --version`)
- Python version (via `python3 --version`)
- Available dependencies
```

### Installation

See [`INSTALL.md`](./INSTALL.md) for complete setup instructions.

Quick setup:
```bash
# Run setup script
python tools/setup/install_tools.py

# Verify installation
python tools/setup/verify_tools.py
```

## Tool Discovery

### Finding the Right Tool

**Decision Tree**:
```text
What do you need?
├── Create new paper structure → paper-scaffold/scaffold.py
├── Generate test data → test-utils/data_generators.mojo
├── Create test fixtures → test-utils/fixtures.mojo
├── Measure performance → benchmarking/
│   ├── Model inference → model_bench.mojo
│   ├── Training speed → training_bench.mojo
│   └── Memory usage → memory_tracker.mojo
└── Generate code → codegen/
    ├── Mojo structs → mojo_boilerplate.py
    ├── Training loops → training_template.py
    └── Data pipelines → data_pipeline.py
```

### Tool Catalog

See [`CATALOG.md`](./CATALOG.md) for complete tool listing with examples.

## Best Practices

### 1. Tool Selection

**DO**:
- Use tools for repetitive tasks (scaffolding, code generation)
- Use tools for performance measurement (benchmarking)
- Use tools for test data generation (consistent, reproducible)

**DON'T**:
- Use tools for one-off tasks (write code directly)
- Over-complicate simple tasks (KISS principle)
- Rely on tools for critical logic (implement directly in Mojo)

### 2. Integration Guidelines

**When creating new tools**:
- Follow ADR-001 language selection
- Document integration points
- Provide usage examples
- Add to tool catalog
- Update this integration guide

**When modifying workflows**:
- Consider tool integration opportunities
- Update documentation
- Test integration end-to-end
- Document in issue-specific README

### 3. Maintenance

**Tool Health Checks**:
```bash
# Verify all tools are functional
python tools/setup/verify_tools.py --verbose

# Check for dependency updates
python tools/setup/check_dependencies.py
```

**Quarterly Reviews**:
- Assess tool usage (which tools are actually used?)
- Update dependencies
- Review Python tools for Mojo conversion
- Archive unused tools

## Troubleshooting

### Tool Not Found

```bash
# Ensure you're in repository root
cd /path/to/ml-odyssey

# Verify tool exists
ls tools/paper-scaffold/scaffold.py
```

### Import Errors

```bash
# Check Mojo path configuration
export MOJO_PATH=/path/to/ml-odyssey

# Verify imports in Mojo REPL
mojo
>>> from tools.test_utils import generate_batch
```

### Permission Issues

```bash
# Make scripts executable
chmod +x tools/*/\*.py

# Or run with Python explicitly
python3 tools/paper-scaffold/scaffold.py
```

## Examples

### Complete Paper Implementation Workflow

```bash
# 1. Scaffold new paper
python tools/paper-scaffold/scaffold.py \
    --paper "ResNet" \
    --output papers/resnet/

# 2. Generate layer implementations
python tools/codegen/mojo_boilerplate.py \
    --type layer \
    --name ResidualBlock \
    --output papers/resnet/layers.mojo

# 3. Implement model (manual work)
# Edit papers/resnet/model.mojo

# 4. Generate training loop
python tools/codegen/training_template.py \
    --optimizer SGD \
    --output papers/resnet/train.mojo

# 5. Write tests with fixtures
# Use tools/test-utils/ in test files

# 6. Run benchmarks
mojo tools/benchmarking/model_bench.mojo \
    --paper resnet

# 7. Generate report
python tools/benchmarking/report_generator.py \
    --paper resnet
```

## References

- [Tools README](./README.md) - Main tools documentation
- [Scripts README](../scripts/README.md) - Automation scripts
- [ADR-001](../notes/review/adr/ADR-001-language-selection-tooling.md) - Language selection
- [Agent Hierarchy](../agents/hierarchy.md) - Agent system overview
- [CLAUDE.md](../CLAUDE.md) - Project guidelines

---

**Document**: `/tools/INTEGRATION.md`
**Purpose**: Comprehensive integration guide for tools directory
**Audience**: Developers, maintainers, agents
**Status**: Living document (update as tools and workflows evolve)
