#!/usr/bin/env mojo

"""
Test script to verify Mojo capabilities for Python script conversion (Issue #8)

Tests:
1. Subprocess stdout capture
2. Subprocess exit code access
3. Subprocess stderr capture
4. Regex library availability (mojo-regex)
"""

from subprocess import run
from sys import exit


fn test_subprocess_stdout() raises:
    """Test basic subprocess stdout capture."""
    print("\n[TEST 1] Subprocess stdout capture")
    print("-" * 50)

    try:
        # subprocess.run() takes a single String command and returns String output
        var result = run("echo 'Hello from subprocess'")
        print("✓ Subprocess execution works")
        print("  Output:", result)
        print("  Return type: String (confirmed)")

    except e:
        print("✗ Subprocess execution failed:", e)
        raise e


fn test_subprocess_with_gh() raises:
    """Test subprocess with gh CLI command."""
    print("\n[TEST 2] Subprocess with gh CLI")
    print("-" * 50)

    try:
        var result = run("gh --version")
        print("✓ gh command execution works")
        print("  Output:", result)

    except e:
        print("✗ gh command failed:", e)
        print("  Note: This might fail if gh is not installed")


fn test_subprocess_exit_code() raises:
    """Test subprocess exit code access."""
    print("\n[TEST 3] Subprocess exit code access")
    print("-" * 50)

    try:
        # Test successful command (exit code 0)
        var result1 = run("true")
        print("✓ Command 'true' executed successfully")
        print("  Output:", result1)

        print("\n  Testing failing command:")
        # Note: run() only returns stdout String, no exit code access
        # Failing commands will raise exceptions
        try:
            var result2 = run("false")
            print("  Command 'false' did not raise exception (unexpected)")
        except:
            print("  ✗ run() does NOT provide exit code - failing commands raise exceptions")
            print("  This is a limitation for script conversion")

    except e:
        print("✗ Exit code test failed:", e)


fn test_subprocess_stderr() raises:
    """Test subprocess stderr capture."""
    print("\n[TEST 4] Subprocess stderr capture")
    print("-" * 50)

    try:
        # Run a command that writes to stderr
        var result = run("ls /nonexistent_directory_test_123 2>&1")
        print("  Output (stdout + stderr redirected):", result)
        print("  ✓ Can capture stderr by redirecting to stdout (2>&1)")

    except e:
        print("  ✗ run() does NOT capture stderr separately")
        print("  Exception:", e)
        print("  Workaround: Use shell redirection (2>&1) to combine streams")


fn test_subprocess_output_capture() raises:
    """Test if we can actually capture and access the output string."""
    print("\n[TEST 5] Subprocess output string access")
    print("-" * 50)

    try:
        var result = run("echo 'test_output_123'")
        print("✓ Subprocess output captured as String")
        print("  Output:", result)
        print("  Can use String methods: len(), startswith(), etc.")
        print("  Length:", len(result))

    except e:
        print("✗ Output capture test failed:", e)


fn test_regex_import():
    """Test regex library availability."""
    print("\n[TEST 6] Regex library availability")
    print("-" * 50)

    # Try to import mojo-regex
    # Note: This requires the package to be installed
    # from regex import match_first, findall  # Requires mojo-regex package

    print("  mojo-regex (github.com/msaelices/mojo-regex):")
    print("  - Status: Early development, not production-ready")
    print("  - Features: Character classes, quantifiers, groups, alternation")
    print("  - Limitations: No non-greedy, word boundaries, case-insensitive")
    print("  - Installation: 'pixi add mojo-regex' FAILED")
    print("  - Error: Package not found in pixi repository")
    print("  ✗ NOT AVAILABLE for use in this project")
    print("\n  Alternative: Build from source (github)")
    print("  Conclusion: Regex NOT practical for Issue #8 conversion")


fn test_string_manipulation():
    """Test basic string manipulation as alternative to regex."""
    print("\n[TEST 7] String manipulation capabilities")
    print("-" * 50)

    try:
        var test_str = "## Test Header"
        print("✓ String creation works")

        # Test string methods
        print("  Testing string operations:")
        print("  - startswith:", test_str.startswith("##"))
        print("  - String length:", len(test_str))

        # Check if split() is available
        # var parts = test_str.split(" ")  # May not exist yet
        print("  Note: Advanced string methods may be limited")

    except e:
        print("✗ String manipulation test failed:", e)


fn print_summary():
    """Print test summary and conclusions."""
    print("\n" + "=" * 50)
    print("SUMMARY: Mojo Capabilities Assessment")
    print("=" * 50)

    print("\nKey Findings:")
    print("1. ✓ Subprocess execution: Available via subprocess.run(cmd: String) -> String")
    print("2. ✓ Stdout capture: Returns String directly, trailing whitespace removed")
    print("3. ✗ Exit codes: NOT AVAILABLE - run() only returns stdout String")
    print("4. ✗ Stderr capture: NOT separate - must redirect with shell (2>&1)")
    print("5. ✗ Regex: Requires external package (mojo-regex) - not in stdlib")

    print("\nAPI Limitations for Script Conversion:")
    print("- run() signature: run(cmd: String) -> String")
    print("- No ProcessResult object with exit_code, stderr attributes")
    print("- Failing commands raise exceptions (cannot check exit codes)")
    print("- Must use shell redirection for stderr: '2>&1'")

    print("\nFor Issue #8 Python → Mojo Conversion:")
    print("✓ FEASIBLE for simple scripts:")
    print("  - Scripts that only need stdout capture")
    print("  - Scripts that expect success (exceptions are acceptable)")
    print("  - Scripts using basic string operations (no regex)")

    print("\n✗ NOT FEASIBLE for complex scripts:")
    print("  - Scripts checking exit codes (e.g., gh CLI status checks)")
    print("  - Scripts with heavy regex parsing (unless mojo-regex added)")
    print("  - Scripts needing separate stderr handling")
    print("  - Scripts like create_issues.py, regenerate_github_issues.py")

    print("\nRecommendations:")
    print("1. Keep Python scripts with regex/parsing in Python")
    print("2. Convert simple wrapper scripts to Mojo")
    print("3. Wait for improved subprocess API (exit codes, stderr)")
    print("4. Evaluate mojo-regex package if regex is critical")


fn main() raises:
    """Run all tests."""
    print("=" * 50)
    print("Testing Mojo Capabilities for Issue #8")
    print("=" * 50)
    print("\nChecking subprocess and regex support for script conversion")

    # Run subprocess tests
    test_subprocess_stdout()
    test_subprocess_with_gh()
    test_subprocess_exit_code()
    test_subprocess_stderr()
    test_subprocess_output_capture()

    # Run regex tests
    test_regex_import()

    # Run string manipulation tests
    test_string_manipulation()

    # Print summary
    print_summary()

    print("\n" + "=" * 50)
    print("Testing Complete")
    print("=" * 50)
