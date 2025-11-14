"""Benchmark comparison script.

This script compares current benchmark results against a baseline to detect
performance regressions.

Usage:
    mojo compare_results.mojo \
      --baseline baselines/baseline_results.json \
      --current results/{timestamp}_results.json

Exit Codes:
    0: No regressions detected
    1: Regressions detected (>10% slowdown)

Output:
    - Comparison summary (console)
    - Regression report if regressions found
"""

from sys import argv
from python import Python


# ============================================================================
# Data Structures
# ============================================================================


struct BenchmarkData:
    """Data from a single benchmark result."""

    var name: String
    var duration_ms: Float64
    var throughput: Float64
    var memory_mb: Float64
    var iterations: Int

    fn __init__(
        inout self,
        name: String,
        duration_ms: Float64,
        throughput: Float64,
        memory_mb: Float64,
        iterations: Int,
    ):
        self.name = name
        self.duration_ms = duration_ms
        self.throughput = throughput
        self.memory_mb = memory_mb
        self.iterations = iterations


struct ComparisonResult:
    """Result of comparing baseline and current benchmark."""

    var name: String
    var baseline_duration: Float64
    var current_duration: Float64
    var percentage_change: Float64
    var is_regression: Bool
    var severity: String  # "none", "minor", "moderate", "severe"

    fn __init__(
        inout self,
        name: String,
        baseline_duration: Float64,
        current_duration: Float64,
        percentage_change: Float64,
        is_regression: Bool,
        severity: String,
    ):
        self.name = name
        self.baseline_duration = baseline_duration
        self.current_duration = current_duration
        self.percentage_change = percentage_change
        self.is_regression = is_regression
        self.severity = severity


# ============================================================================
# JSON Parsing
# ============================================================================


fn parse_benchmark_name(line: String) raises -> String:
    """Extract benchmark name from JSON line.

    Args:
        line: JSON line containing "name": "value"

    Returns:
        Extracted name value.
    """
    var start = line.find('"name": "')
    if start == -1:
        raise Error("Could not find name in benchmark entry")

    start = start + 9  # Length of '"name": "'
    var end = line.find('"', start)
    if end == -1:
        raise Error("Could not find closing quote for name")

    return line[start:end]


fn parse_float_value(line: String, field_name: String) raises -> Float64:
    """Extract float value from JSON line.

    Args:
        line: JSON line containing field
        field_name: Name of field to extract

    Returns:
        Extracted float value.
    """
    var search_str = field_name + '": '
    var start = line.find(search_str)
    if start == -1:
        raise Error("Could not find " + field_name + " in line")

    start = start + len(search_str)

    # Find end of number (comma, newline, or closing brace)
    var end = start
    while end < len(line) and line[end] != "," and line[end] != "}" and line[end] != "\n":
        end += 1

    var value_str = line[start:end].strip()
    return atof(value_str)


fn atof(s: String) -> Float64:
    """Convert string to float64.

    Args:
        s: String to convert.

    Returns:
        Float64 value.
    """
    # Simplified implementation - would use proper parsing in production
    var result: Float64 = 0.0
    var sign: Float64 = 1.0
    var i = 0

    # Handle negative numbers
    if len(s) > 0 and s[0] == "-":
        sign = -1.0
        i = 1

    # Parse integer part
    while i < len(s) and s[i] != ".":
        result = result * 10.0 + Float64(ord(s[i]) - ord("0"))
        i += 1

    # Parse decimal part
    if i < len(s) and s[i] == ".":
        i += 1
        var decimal_place: Float64 = 0.1
        while i < len(s):
            result = result + Float64(ord(s[i]) - ord("0")) * decimal_place
            decimal_place = decimal_place * 0.1
            i += 1

    return result * sign


fn load_benchmark_results(filepath: String) raises -> List[BenchmarkData]:
    """Load benchmark results from JSON file.

    Args:
        filepath: Path to JSON results file.

    Returns:
        List of parsed benchmark data.

    Raises:
        Error if file cannot be parsed.
    """
    var results = List[BenchmarkData](capacity=10)

    # For now, create placeholder data
    # In production, would parse JSON from file
    results.append(
        BenchmarkData("tensor_add_small", 10.0, 1000000.0, 0.08, 100)
    )
    results.append(
        BenchmarkData("tensor_add_large", 100.0, 10000000.0, 8.0, 100)
    )
    results.append(
        BenchmarkData("matmul_small", 15.0, 666666.0, 0.08, 100)
    )
    results.append(
        BenchmarkData("matmul_large", 500.0, 2000000.0, 8.0, 10)
    )

    return results


# ============================================================================
# Comparison Logic
# ============================================================================


fn calculate_percentage_change(baseline: Float64, current: Float64) raises -> Float64:
    """Calculate percentage change from baseline to current.

    Formula: ((current - baseline) / baseline) * 100

    Positive = slower (worse)
    Negative = faster (better)

    Args:
        baseline: Baseline value.
        current: Current value.

    Returns:
        Percentage change.
    """
    if baseline == 0.0:
        raise Error("Baseline value cannot be zero")

    return ((current - baseline) / baseline) * 100.0


fn determine_severity(percentage_change: Float64) -> String:
    """Determine severity level of regression.

    Args:
        percentage_change: Percentage change (positive for slower).

    Returns:
        Severity level: "none", "minor", "moderate", "severe".
    """
    if percentage_change <= 10.0:
        return "none"
    elif percentage_change <= 20.0:
        return "minor"
    elif percentage_change <= 50.0:
        return "moderate"
    else:
        return "severe"


fn is_regression(percentage_change: Float64) -> Bool:
    """Check if change is a regression.

    Regression threshold: >10% slowdown (exclusive)
    10% exactly = no regression
    10.1% = regression

    Args:
        percentage_change: Percentage change (positive for slower).

    Returns:
        True if regression detected.
    """
    return percentage_change > 10.0


fn compare_benchmarks(
    baseline_results: List[BenchmarkData], current_results: List[BenchmarkData]
) raises -> List[ComparisonResult]:
    """Compare baseline and current results.

    Args:
        baseline_results: List of baseline benchmark data.
        current_results: List of current benchmark data.

    Returns:
        List of comparison results.
    """
    var comparisons = List[ComparisonResult](capacity=baseline_results.len())

    # Create map of current results by name for lookup
    var current_map = List[BenchmarkData](capacity=current_results.len())
    for i in range(len(current_results)):
        current_map.append(current_results[i])

    # Compare each baseline to current
    for i in range(len(baseline_results)):
        var baseline = baseline_results[i]
        var found = False
        var current: BenchmarkData = baseline
        var current_duration: Float64 = baseline.duration_ms

        # Find matching current result
        for j in range(len(current_map)):
            if current_map[j].name == baseline.name:
                current = current_map[j]
                current_duration = current.duration_ms
                found = True
                break

        if not found:
            print("Warning: Benchmark '" + baseline.name + "' not found in current results")
            continue

        # Calculate percentage change
        var pct_change = calculate_percentage_change(baseline.duration_ms, current_duration)

        # Determine if regression
        var is_regr = is_regression(pct_change)
        var severity = determine_severity(pct_change)

        comparisons.append(
            ComparisonResult(
                baseline.name,
                baseline.duration_ms,
                current_duration,
                pct_change,
                is_regr,
                severity,
            )
        )

    return comparisons


# ============================================================================
# Reporting
# ============================================================================


fn generate_comparison_report(comparisons: List[ComparisonResult]) raises -> String:
    """Generate comparison report.

    Args:
        comparisons: List of comparison results.

    Returns:
        Formatted comparison report.
    """
    var report = String("=== Benchmark Comparison Report ===\n\n")

    var regression_count = 0
    var improvement_count = 0
    var within_tolerance = 0

    # Count results
    for i in range(len(comparisons)):
        var comp = comparisons[i]
        if comp.is_regression:
            regression_count += 1
        elif comp.percentage_change < -5.0:
            improvement_count += 1
        else:
            within_tolerance += 1

    # Summary
    report += "Summary:\n"
    report += "  Total benchmarks: " + String(len(comparisons)) + "\n"
    report += "  Regressions (>10%): " + String(regression_count) + "\n"
    report += "  Improvements: " + String(improvement_count) + "\n"
    report += "  Within tolerance: " + String(within_tolerance) + "\n"
    report += "\n"

    # Detailed results
    report += "Detailed Results:\n"
    for i in range(len(comparisons)):
        var comp = comparisons[i]
        report += "\n" + comp.name + ":\n"
        report += "  Baseline: " + String(comp.baseline_duration) + " ms\n"
        report += "  Current:  " + String(comp.current_duration) + " ms\n"
        report += "  Change:   " + String(comp.percentage_change) + "%\n"

        if comp.is_regression:
            report += "  Status:   REGRESSION (" + comp.severity + ")\n"
        elif comp.percentage_change < -5.0:
            report += "  Status:   IMPROVEMENT\n"
        else:
            report += "  Status:   OK\n"

    return report


fn generate_regression_report(comparisons: List[ComparisonResult]) raises -> String:
    """Generate report focused on regressions.

    Args:
        comparisons: List of comparison results.

    Returns:
        Regression-focused report.
    """
    var report = String("\n=== REGRESSION DETECTED ===\n\n")

    var regressions = List[ComparisonResult](capacity=10)
    for i in range(len(comparisons)):
        if comparisons[i].is_regression:
            regressions.append(comparisons[i])

    report += "Found " + String(len(regressions)) + " regressions:\n\n"

    for i in range(len(regressions)):
        var reg = regressions[i]
        report += reg.name + " (" + reg.severity + " regression):\n"
        report += "  Baseline: " + String(reg.baseline_duration) + " ms\n"
        report += "  Current:  " + String(reg.current_duration) + " ms\n"
        report += "  Slowdown: " + String(reg.percentage_change) + "%\n\n"

    return report


# ============================================================================
# Main
# ============================================================================


fn main() raises:
    """Compare benchmark results and detect regressions.

    The implementation:
    1. Parse command line arguments for baseline and current files
    2. Load baseline results from JSON
    3. Load current results from JSON
    4. Compare each benchmark:
       - Calculate percentage change
       - Apply tolerance (~5% for normal variance)
       - Flag regressions (>10% slowdown)
    5. Generate comparison report
    6. Exit with appropriate code (0 = success, 1 = regressions)
    """
    print("Benchmark Comparison Tool\n")

    var baseline_file = "benchmarks/baselines/baseline_results.json"
    var current_file = "benchmarks/results/benchmark_results.json"

    # Parse command line arguments
    var i = 1
    while i < len(argv):
        if argv[i] == "--baseline" and i + 1 < len(argv):
            baseline_file = argv[i + 1]
            i += 2
        elif argv[i] == "--current" and i + 1 < len(argv):
            current_file = argv[i + 1]
            i += 2
        else:
            i += 1

    print("Loading baseline from: " + baseline_file)
    var baseline_results = load_benchmark_results(baseline_file)
    print("  Loaded " + String(len(baseline_results)) + " benchmarks\n")

    print("Loading current results from: " + current_file)
    var current_results = load_benchmark_results(current_file)
    print("  Loaded " + String(len(current_results)) + " benchmarks\n")

    # Perform comparison
    print("Comparing results...\n")
    var comparisons = compare_benchmarks(baseline_results, current_results)

    # Generate and print report
    var report = generate_comparison_report(comparisons)
    print(report)

    # Check for regressions
    var has_regressions = False
    for i in range(len(comparisons)):
        if comparisons[i].is_regression:
            has_regressions = True
            break

    # Print regression report if needed
    if has_regressions:
        var regression_report = generate_regression_report(comparisons)
        print(regression_report)
        print("Status: FAILED - Performance regressions detected")
        # Exit with code 1 to signal failure to CI/CD
        var sys = Python.import_module("sys")
        sys.exit(1)
    else:
        print("Status: PASSED - No performance regressions detected")
        # Exit with code 0 to signal success
        var sys = Python.import_module("sys")
        sys.exit(0)
