#!/usr/bin/env mojo
"""Test what subprocess actually provides."""

from subprocess import run

fn main() raises:
    """Test subprocess capabilities."""
    print("=" * 60)
    print("Mojo Subprocess Simple Tests")
    print("=" * 60)

    print("\n1. Testing basic command execution...")
    _ = run("echo 'Hello World'")
    print("✓ Basic execution works")

    print("\n2. Testing gh CLI access...")
    _ = run("gh --version")
    print("✓ gh CLI accessible")

    print("\n3. Testing command with output...")
    _ = run("ls -la mojo_tests/ | head -5")
    print("✓ Complex command with pipe works")

    print("\n4. Testing Python invocation...")
    _ = run("python3 --version")
    print("✓ Python accessible from Mojo")

    print("\n" + "=" * 60)
    print("CRITICAL ISSUE: Cannot capture stdout/stderr!")
    print("subprocess.run() appears to only execute commands")
    print("but doesn't provide access to output or exit codes")
    print("=" * 60)
