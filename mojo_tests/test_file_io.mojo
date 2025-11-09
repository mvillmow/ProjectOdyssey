#!/usr/bin/env mojo
"""Test Mojo's file I/O capabilities for script conversion feasibility."""

fn test_read_file() raises:
    """Test reading a file."""
    print("Testing file reading...")

    # Try to read this file itself
    var file_path = "mojo_tests/test_file_io.mojo"

    try:
        var file = open(file_path, "r")
        var content = file.read()
        file.close()

        print("✓ Successfully read file")
        print("  File size:", len(content), "bytes")
        print("  First 100 chars:", content[:100])
        return
    except e:
        print("✗ Failed to read file:", e)
        raise e

fn test_write_file() raises:
    """Test writing to a file."""
    print("\nTesting file writing...")

    var test_file = "mojo_tests/test_output.txt"
    var test_content = "Hello from Mojo!\nThis is a test file.\n"

    try:
        var file = open(test_file, "w")
        file.write(test_content)
        file.close()

        print("✓ Successfully wrote file")

        # Verify by reading back
        var read_file = open(test_file, "r")
        var read_content = read_file.read()
        read_file.close()

        if read_content == test_content:
            print("✓ File content verified")
        else:
            print("✗ File content mismatch")
            print("  Expected:", test_content)
            print("  Got:", read_content)
    except e:
        print("✗ Failed to write file:", e)
        raise e

fn test_path_operations() raises:
    """Test path manipulation."""
    print("\nTesting path operations...")

    # Test basic path construction
    var base_path = "notes/plan"
    var sub_path = "01-foundation"
    var full_path = base_path + "/" + sub_path

    print("✓ Path construction works")
    print("  Result:", full_path)

fn main() raises:
    """Run all file I/O tests."""
    print("=" * 60)
    print("Mojo File I/O Capability Tests")
    print("=" * 60)

    test_read_file()
    test_write_file()
    test_path_operations()

    print("\n" + "=" * 60)
    print("File I/O tests completed!")
    print("=" * 60)
