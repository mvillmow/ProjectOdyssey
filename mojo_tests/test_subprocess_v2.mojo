#!/usr/bin/env mojo
"""Test subprocess with string command."""

from subprocess import run

fn test_subprocess_string() raises:
    """Test subprocess with string command."""
    print("Testing subprocess.run() with string...")

    try:
        # Try with just a string
        var result = run("echo Hello")
        print("✓ subprocess.run() with string works!")
    except e:
        print("✗ subprocess.run() failed:", e)
        raise e

fn main() raises:
    """Test subprocess capabilities."""
    print("=" * 60)
    print("Mojo Subprocess Module Test V2")
    print("=" * 60)

    test_subprocess_string()

    print("\n" + "=" * 60)
    print("Subprocess test completed!")
    print("=" * 60)
