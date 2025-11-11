# Benchmark Workflow

## Overview

Create a GitHub Actions workflow that runs performance benchmarks on paper implementations and compares results against baseline measurements to detect performance regressions.

## Parent Plan
[Parent](../plan.md)

## Child Plans
- [01-run-benchmarks](./01-run-benchmarks/plan.md)
- [02-compare-baseline](./02-compare-baseline/plan.md)
- [03-publish-results](./03-publish-results/plan.md)

## Inputs
- Run benchmarks automatically on relevant code changes
- Compare benchmark results against baseline
- Detect performance regressions early
- Track performance trends over time
- Publish benchmark results for visibility

## Outputs
- Completed benchmark workflow
- Run benchmarks automatically on relevant code changes (completed)

## Steps
1. Run Benchmarks
2. Compare Baseline
3. Publish Results

## Success Criteria
- [ ] Benchmarks run on performance-critical changes
- [ ] Results compared against baseline accurately
- [ ] Regressions detected and reported
- [ ] Results published to PR comments or artifacts
- [ ] Historical trends tracked
- [ ] Benchmark execution time reasonable (under 20 minutes)

## Notes
- Only run on changes to core implementation files
- Use consistent environment for fair comparisons
- Store baseline results in repository or external storage
- Consider using benchmarking frameworks like criterion
- Allow some variance in results (e.g., 5% tolerance)