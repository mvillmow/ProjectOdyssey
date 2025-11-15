# Benchmark Tooling Installation Guide

## Overview

This guide explains how to install and use the ML Odyssey benchmarking infrastructure.

## Distribution Package

**Version**: 0.1.0

**Package**: `benchmarks-0.1.0.tar.gz`

**Contents**:

- Benchmark execution scripts (Mojo)
- Baseline comparison tools
- Sample baseline results
- Documentation
- License

## Installation

### From Distribution Archive

1. Extract the archive:

```bash
tar -xzf benchmarks-0.1.0.tar.gz
```

This will create a `benchmarks/` directory with the following structure:

```text
benchmarks/
├── README.md                      # Comprehensive documentation
├── baselines/                     # Baseline results
│   └── baseline_results.json      # Reference baseline
├── scripts/                       # Benchmark execution scripts
│   ├── run_benchmarks.mojo        # Main benchmark runner
│   └── compare_results.mojo       # Baseline comparison tool
└── results/                       # Results directory (initially empty)
```

1. Verify installation:

```bash
cd benchmarks
ls -la scripts/
```

### System Requirements

- **OS**: Linux, macOS (Intel/Apple Silicon), Windows (WSL2)
- **Memory**: 4GB minimum, 8GB recommended
- **Disk Space**: 500MB for installation + space for results
- **Bash**: Version 4.0+ required

### Prerequisites

- Mojo compiler (v0.25.7 or later)
- Python 3.7+ (for JSON result processing)
- Sufficient disk space for results storage

## Usage

### Running Benchmarks

Execute all benchmarks:

```bash
mojo benchmarks/scripts/run_benchmarks.mojo
```

Output will be written to:

```text
benchmarks/results/{timestamp}_results.json
```

### Comparing Results to Baseline

Compare your results against the baseline:

```bash
mojo benchmarks/scripts/compare_results.mojo \
  --baseline benchmarks/baselines/baseline_results.json \
  --current benchmarks/results/{timestamp}_results.json
```

**Exit codes**:

- `0` - No performance regressions detected
- `1` - Performance regressions detected (>10% slowdown)

**Regression severity levels**:

- **Minor**: 10-20% slowdown (warning)
- **Moderate**: 20-50% slowdown (requires investigation)
- **Severe**: >50% slowdown (critical issue)

## CI/CD Integration

The distribution includes automated benchmark execution via GitHub Actions.

### Workflow File

Location: `.github/workflows/benchmark.yml`

**Triggers**:

- Manual dispatch (`workflow_dispatch`)
- Scheduled nightly runs (2 AM UTC)
- Pull requests with `benchmark` label

**Features**:

- Matrix strategy for parallel execution
- Baseline comparison
- Regression detection
- PR comments with results
- Artifact retention (90 days)

### Manual Trigger

```bash
# Using GitHub CLI
gh workflow run benchmark.yml

# Or via GitHub web interface
# Actions → Performance Benchmarks → Run workflow
```

### PR Integration

Add the `benchmark` label to any PR to trigger benchmarks:

```bash
gh pr edit <PR_NUMBER> --add-label benchmark
```

## Benchmark Suites

The workflow supports three benchmark suites:

1. **tensor-ops** - Tensor operation benchmarks
2. **model-training** - Training loop benchmarks
3. **data-loading** - Data pipeline benchmarks

Run specific suite:

```bash
gh workflow run benchmark.yml -f suite=tensor-ops
```

## Verification

### Test Benchmark Execution

```bash
cd benchmarks
mojo scripts/run_benchmarks.mojo
```

Expected output:

```text
Running benchmarks...
- tensor_add_small: XX.X ms
- tensor_add_large: XX.X ms
- matmul_small: XX.X ms
- matmul_large: XX.X ms

Results written to: results/{timestamp}_results.json
```

### Test Baseline Comparison

```bash
mojo scripts/compare_results.mojo \
  --baseline baselines/baseline_results.json \
  --current results/{latest_timestamp}_results.json
```

Expected output:

```text
Comparing results...
✓ tensor_add_small: 0.5% faster
✓ tensor_add_large: 1.2% slower
✓ matmul_small: 0.8% faster
✓ matmul_large: 2.1% slower

No regressions detected
```

## Troubleshooting

### Issue: Benchmark fails to run

**Solution**:

1. Verify Mojo compiler is installed: `mojo --version`
1. Check file permissions: `chmod +x scripts/*.mojo`
1. Ensure all dependencies are available

### Issue: Comparison fails

**Solution**:

1. Verify baseline file exists: `ls baselines/baseline_results.json`
1. Check result file exists: `ls results/*.json`
1. Validate JSON syntax: `python3 -m json.tool < results/file.json`

### Issue: CI workflow doesn't run

**Solution**:

1. Verify workflow file is in `.github/workflows/`
1. Check workflow triggers are configured correctly
1. Ensure GitHub Actions are enabled for repository
1. Verify `benchmark` label is applied to PR

## Updating Baselines

Baseline results should only be updated after verification:

1. Run benchmarks and verify results are expected
1. Copy new results to baselines:

```bash
cp results/{timestamp}_results.json baselines/baseline_results.json
```

1. Commit updated baseline:

```bash
git add baselines/baseline_results.json
git commit -m "chore(benchmarks): update baseline results"
```

1. Document reason for baseline update in commit message

## Uninstallation

To remove the benchmark tooling:

```bash
rm -rf benchmarks/
```

To remove CI/CD integration:

```bash
rm .github/workflows/benchmark.yml
```

## Support

For issues, questions, or contributions:

- **Documentation**: See `benchmarks/README.md`
- **Issue Tracking**: GitHub Issues
- **Planning**: Issue #52 ([Plan] Benchmarks)
- **Testing**: Issue #53 ([Test] Benchmarks)
- **Implementation**: Issue #54 ([Impl] Benchmarks)
- **Packaging**: Issue #55 ([Package] Benchmarks)

## License

BSD 3-Clause License - See LICENSE file for details

## Version History

- **0.1.0** (2025-01-14): Initial distribution release
  - Benchmark runner implementation
  - Baseline comparison tool
  - CI/CD workflow integration
  - Placeholder baseline values (to be updated with real benchmarks)
