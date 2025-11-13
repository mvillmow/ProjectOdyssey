"""Benchmarking infrastructure test suite.

This package contains tests for the benchmarking infrastructure including:
- Benchmark execution and timing
- Baseline loading and management
- Result comparison logic
- Regression detection
- CI/CD integration

Test Structure:
    test_benchmark_runner.mojo - Benchmark execution, timing, iterations
    test_baseline_loader.mojo - Loading baseline data from JSON
    test_result_comparison.mojo - Comparison logic, threshold detection
    test_regression_detection.mojo - >10% slowdown alerts
    test_ci_integration.mojo - CI/CD workflow integration

All tests follow TDD principles:
- Test behavior, not implementation
- Use shared fixtures from tests/shared/conftest.mojo
- Deterministic and reproducible
- Focus on critical functionality
"""
