#!/usr/bin/env python3
"""
Validate Test Coverage

Ensures all test_*.mojo files are included in the CI workflow matrix.
Prevents silently skipped tests by validating comprehensive-tests.yml coverage.

Usage:
    python scripts/validate_test_coverage.py
"""

import sys
from pathlib import Path
import yaml


def find_all_test_files():
    """Find all test_*.mojo files in the repository."""
    test_files = []
    test_root = Path("tests")

    if not test_root.exists():
        print("⚠️  tests/ directory not found")
        return test_files

    for test_file in test_root.rglob("test_*.mojo"):
        test_files.append(test_file)

    return sorted(test_files)


def load_ci_workflow():
    """Load the comprehensive-tests.yml workflow file."""
    workflow_path = Path(".github/workflows/comprehensive-tests.yml")

    if not workflow_path.exists():
        print(f"❌ Workflow file not found: {workflow_path}")
        sys.exit(1)

    with open(workflow_path) as f:
        return yaml.safe_load(f)


def extract_ci_test_groups(workflow):
    """Extract test groups from the CI workflow matrix."""
    try:
        jobs = workflow.get("jobs", {})
        test_job = jobs.get("test-mojo-comprehensive", {})
        strategy = test_job.get("strategy", {})
        matrix = strategy.get("matrix", {})
        test_groups = matrix.get("test-group", [])

        return test_groups
    except Exception as e:
        print(f"❌ Failed to parse workflow: {e}")
        sys.exit(1)


def validate_coverage(test_files, test_groups):
    """Validate that all test files are covered by CI test groups."""
    uncovered_tests = []
    covered_tests = []

    # Build a mapping of directories to their test groups
    group_paths = {}
    for group in test_groups:
        path = group.get("path", "")
        pattern = group.get("pattern", "")
        group_paths[path] = pattern

    # Check each test file
    for test_file in test_files:
        test_path = str(test_file)
        is_covered = False

        # Check if test file matches any group
        for group_path, pattern in group_paths.items():
            if test_path.startswith(group_path):
                # Simple pattern matching - assumes test_*.mojo pattern
                if "test_" in pattern or pattern == "test_*.mojo":
                    is_covered = True
                    covered_tests.append(test_file)
                    break

        if not is_covered:
            uncovered_tests.append(test_file)

    return covered_tests, uncovered_tests


def main():
    """Main validation logic."""
    print("=" * 60)
    print("Test Coverage Validation")
    print("=" * 60)
    print()

    # Find all test files
    print("Finding all test_*.mojo files...")
    test_files = find_all_test_files()
    print(f"Found {len(test_files)} test files")
    print()

    # Load CI workflow
    print("Loading CI workflow configuration...")
    workflow = load_ci_workflow()
    test_groups = extract_ci_test_groups(workflow)
    print(f"Found {len(test_groups)} test groups in CI workflow")
    print()

    # Validate coverage
    print("Validating test coverage...")
    covered_tests, uncovered_tests = validate_coverage(test_files, test_groups)

    print(f"✅ Covered: {len(covered_tests)} tests")
    print(f"❌ Uncovered: {len(uncovered_tests)} tests")
    print()

    # Report uncovered tests
    if uncovered_tests:
        print("⚠️  WARNING: The following tests are NOT covered by CI:")
        print()
        for test_file in uncovered_tests:
            print(f"  - {test_file}")
        print()
        print("These tests will NOT run in CI and may silently fail.")
        print("Add them to .github/workflows/comprehensive-tests.yml")
        print()

        # Create validation report
        with open("coverage-validation.txt", "w") as f:
            f.write("Test Coverage Validation Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total Tests: {len(test_files)}\n")
            f.write(f"Covered: {len(covered_tests)}\n")
            f.write(f"Uncovered: {len(uncovered_tests)}\n\n")

            if uncovered_tests:
                f.write("Uncovered Tests:\n")
                for test_file in uncovered_tests:
                    f.write(f"  - {test_file}\n")

        sys.exit(1)
    else:
        print("✅ All tests are covered by CI workflow!")
        print()

        # Create validation report
        with open("coverage-validation.txt", "w") as f:
            f.write("Test Coverage Validation Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total Tests: {len(test_files)}\n")
            f.write(f"Covered: {len(covered_tests)}\n")
            f.write("Uncovered: 0\n\n")
            f.write("✅ All tests are covered by CI workflow!\n")

        sys.exit(0)


if __name__ == "__main__":
    main()
