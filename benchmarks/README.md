# Benchmarking Infrastructure

## Overview

This directory contains the benchmarking infrastructure for ML Odyssey, enabling performance testing,
baseline comparison, regression detection, and CI/CD integration.

## Architecture

### 3-Tier Architecture

#### Tier 1: Benchmarks Directory

Performance testing infrastructure with:

- Benchmark scripts for execution
- Results storage with timestamps
- Reproducible testing environment

#### Tier 2: Benchmark Validator Tool

Baseline comparison system that:

- Loads baseline benchmark data
- Compares current results to baseline
- Detects performance regressions

#### Tier 3: CI/CD Integration

Automated execution that:

- Runs benchmarks on PR checks
- Compares against baselines
- Tracks historical performance

## Directory Structure

```text
benchmarks/
├── README.md              # This file
├── baselines/             # Baseline benchmark results
│   └── baseline_results.json
├── scripts/               # Benchmark execution scripts
│   ├── run_benchmarks.mojo
│   └── compare_results.mojo
└── results/               # Timestamped results
    └── {timestamp}_results.json
```

## Performance Targets

- **Execution Time**: < 15 minutes for full benchmark suite
- **Iterations**: Multiple iterations for statistical validity
- **Normal Variance**: ~5% tolerance for performance variation
- **Regression Alert**: >10% slowdown triggers alert

## Usage

### Running Benchmarks

```bash
# Run all benchmarks
mojo benchmarks/scripts/run_benchmarks.mojo

# Output: benchmarks/results/{timestamp}_results.json
```

### Comparing Results

```bash
# Compare latest results to baseline
mojo benchmarks/scripts/compare_results.mojo \
  --baseline benchmarks/baselines/baseline_results.json \
  --current benchmarks/results/{timestamp}_results.json

# Exit code 0: No regressions
# Exit code 1: Regressions detected (>10% slowdown)
```

### CI/CD Integration

Benchmarks run automatically on:

- Pull requests (comparison to baseline)
- Main branch commits (update baseline if needed)
- Scheduled runs (weekly performance tracking)

See `.github/workflows/benchmarks.yml` for CI configuration.

## Result Format

### JSON Structure

```json
{
  "timestamp": "2025-01-13T10:00:00Z",
  "environment": {
    "os": "linux",
    "cpu": "x86_64",
    "mojo_version": "0.25.7"
  },
  "benchmarks": [
    {
      "name": "tensor_add",
      "duration_ms": 12.5,
      "throughput": 8000000.0,
      "memory_mb": 2.5,
      "iterations": 100
    }
  ]
}
```

### Baseline Management

- **Initial baseline**: Created from first stable benchmark run
- **Baseline updates**: Only on explicit approval after verification
- **Version tracking**: Baselines linked to release versions

## Development

### Adding New Benchmarks

1. Add benchmark to `scripts/run_benchmarks.mojo`
2. Run benchmark suite to verify
3. Update baseline if needed (with approval)
4. Document benchmark in this README

### Benchmark Design Principles

- **Deterministic**: Use fixed random seeds
- **Isolated**: Each benchmark independent
- **Realistic**: Reflect actual usage patterns
- **Fast**: Individual benchmarks < 1 minute
- **Meaningful**: Test critical performance paths

## References

- **Planning**: `notes/issues/52/README.md` (Issue #52 - Plan)
- **Tests**: `tests/tooling/benchmarks/` (Issue #53 - Test)
- **Implementation**: Issue #54
- **CI/CD**: `.github/workflows/benchmarks.yml`
