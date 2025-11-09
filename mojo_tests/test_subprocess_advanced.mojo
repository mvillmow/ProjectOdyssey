#!/usr/bin/env mojo
"""Test advanced subprocess features needed for script conversion."""

from subprocess import run

fn test_capture_output() raises:
    """Test capturing command output."""
    print("Testing output capture...")

    var result = run("echo 'Hello World'")

    # Check what properties are available on result
    print("  Result type:", type(result).__name__)

    # Try to access common attributes
    try:
        print("  Has stdout:", hasattr(result, "stdout"))
        print("  Has stderr:", hasattr(result, "stderr"))
        print("  Has exit_code:", hasattr(result, "exit_code"))
    except:
        print("  Cannot introspect result attributes")

fn test_gh_command() raises:
    """Test running gh CLI command."""
    print("\nTesting GitHub CLI command...")

    try:
        var result = run("gh --version")
        print("✓ gh CLI accessible from Mojo!")
    except e:
        print("✗ Failed to run gh command:", e)

fn test_exit_code() raises:
    """Test checking exit codes."""
    print("\nTesting exit code handling...")

    # Try a command that should succeed
    var result_ok = run("true")
    print("  Command 'true' completed")

    # Try a command that should fail
    var result_fail = run("false")
    print("  Command 'false' completed")

fn main() raises:
    """Test advanced subprocess features."""
    print("=" * 60)
    print("Mojo Advanced Subprocess Tests")
    print("=" * 60)

    test_capture_output()
    test_gh_command()
    test_exit_code()

    print("\n" + "=" * 60)
    print("Advanced subprocess tests completed!")
    print("=" * 60)
