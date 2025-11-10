#!/usr/bin/env mojo

from subprocess import run

fn main() raises:
    """Test if run() raises exception on non-zero exit code."""
    print("Test 1: true command (exit 0)")
    try:
        var result = run("true")
        print("  Success - no exception raised")
        print("  Output: '" + result + "'")
    except e:
        print("  Exception raised:", e)

    print("\nTest 2: false command (exit 1)")
    try:
        var result = run("false")
        print("  Success - no exception raised (surprising!)")
        print("  Output: '" + result + "'")
    except e:
        print("  Exception raised:", e)

    print("\nTest 3: exit 42 command")
    try:
        var result = run("exit 42")
        print("  Success - no exception raised")
        print("  Output: '" + result + "'")
    except e:
        print("  Exception raised:", e)

    print("\nTest 4: non-existent command")
    try:
        var result = run("nonexistent_command_xyz_123")
        print("  Success - no exception raised")
        print("  Output: '" + result + "'")
    except e:
        print("  Exception raised (expected):", e)

    print("\nConclusion:")
    print("  run() does NOT raise exceptions for non-zero exit codes")
    print("  run() ONLY raises for command execution errors")
    print("  This means exit codes are COMPLETELY IGNORED")
