#!/usr/bin/env mojo
"""Test actual subprocess module if available."""

from subprocess import run

fn test_subprocess_module() raises:
    """Test the subprocess module."""
    print("Testing subprocess.run()...")

    try:
        # Try a simple command
        var result = run(["echo", "Hello from subprocess"])
        print("✓ subprocess.run() works!")
        print("  Exit code:", result.exit_code)
    except e:
        print("✗ subprocess.run() failed:", e)
        raise e

fn main() raises:
    """Test subprocess capabilities."""
    print("=" * 60)
    print("Mojo Subprocess Module Test")
    print("=" * 60)

    test_subprocess_module()

    print("\n" + "=" * 60)
    print("Subprocess test completed!")
    print("=" * 60)
