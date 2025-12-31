#!/usr/bin/env python3
"""Test script for git bisect to find heap corruption fix.

This script creates a combined test file with 24 tests and runs it.
Exit codes:
  0 = tests pass (bug is fixed)
  1 = tests crash/fail (bug is present)
  125 = skip this commit (can't build/test)

Usage:
  git bisect start HEAD aee64aea~1
  git bisect run python3 scripts/bisect_heap_test.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Get the repo root - use environment variable or find it from git
def get_repo_root():
    """Get the repository root directory."""
    if "REPO_ROOT" in os.environ:
        return Path(os.environ["REPO_ROOT"])
    # Try to find it from git
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass
    # Fallback to hardcoded path
    return Path("/home/mvillmow/ProjectOdyssey")

REPO_ROOT = get_repo_root()

# Combined test file content - runs 24 tests in one file
TEST_FILE_CONTENT = '''"""Combined test to detect heap corruption bug #2942.

Runs 24 tests in a single file. If heap corruption is present, this will crash.
"""

from shared.core.extensor import ExTensor
from shared.core.conv import conv2d
from shared.core.linear import linear
from shared.testing.layer_params import ConvFixture, LinearFixture
from shared.testing.assertions import assert_shape, assert_dtype, assert_false
from shared.testing.special_values import create_special_value_tensor, SPECIAL_VALUE_ONE
from shared.testing.layer_testers import LayerTester
from math import isnan, isinf


fn create_conv1_params(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    var f = ConvFixture(in_channels=1, out_channels=6, kernel_size=5, dtype=dtype)
    return f.kernel, f.bias

fn create_conv2_params(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    var f = ConvFixture(in_channels=6, out_channels=16, kernel_size=5, dtype=dtype)
    return f.kernel, f.bias

fn create_fc1_params(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    var f = LinearFixture(in_features=400, out_features=120, dtype=dtype)
    return f.weights, f.bias

fn create_fc2_params(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    var f = LinearFixture(in_features=120, out_features=84, dtype=dtype)
    return f.weights, f.bias

fn create_fc3_params(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    var f = LinearFixture(in_features=84, out_features=10, dtype=dtype)
    return f.weights, f.bias


# Conv1 tests
fn test_conv1_forward_float32() raises:
    var r = create_conv1_params(DType.float32)
    LayerTester.test_conv_layer(1, 6, 5, 28, 28, r[0], r[1], DType.float32)

fn test_conv1_forward_float16() raises:
    var r = create_conv1_params(DType.float16)
    LayerTester.test_conv_layer(1, 6, 5, 28, 28, r[0], r[1], DType.float16)

fn test_conv1_backward_float32() raises:
    var r = create_conv1_params(DType.float32)
    LayerTester.test_conv_layer_backward(1, 6, 5, 8, 8, r[0], r[1], DType.float32)

# Conv2 tests
fn test_conv2_forward_float32() raises:
    var r = create_conv2_params(DType.float32)
    LayerTester.test_conv_layer(6, 16, 5, 14, 14, r[0], r[1], DType.float32)

fn test_conv2_forward_float16() raises:
    var r = create_conv2_params(DType.float16)
    LayerTester.test_conv_layer(6, 16, 5, 14, 14, r[0], r[1], DType.float16)

fn test_conv2_backward_float32() raises:
    var r = create_conv2_params(DType.float32)
    LayerTester.test_conv_layer_backward(6, 16, 5, 8, 8, r[0], r[1], DType.float32)

# ReLU tests
fn test_relu_forward_float32() raises:
    var s: List[Int] = [1, 6, 24, 24]
    LayerTester.test_activation_layer(s, DType.float32, activation="relu")

fn test_relu_forward_float16() raises:
    var s: List[Int] = [1, 6, 24, 24]
    LayerTester.test_activation_layer(s, DType.float16, activation="relu")

fn test_relu_backward_float32() raises:
    var s: List[Int] = [1, 6, 8, 8]
    LayerTester.test_activation_layer_backward(s, DType.float32, activation="relu")

# MaxPool tests
fn test_maxpool1_forward_float32() raises:
    LayerTester.test_pooling_layer(6, 24, 24, 2, 2, DType.float32, "max", 0)

fn test_maxpool1_forward_float16() raises:
    LayerTester.test_pooling_layer(6, 24, 24, 2, 2, DType.float16, "max", 0)

fn test_maxpool2_forward_float32() raises:
    LayerTester.test_pooling_layer(16, 10, 10, 2, 2, DType.float32, "max", 0)

fn test_maxpool2_forward_float16() raises:
    LayerTester.test_pooling_layer(16, 10, 10, 2, 2, DType.float16, "max", 0)

# FC1 tests
fn test_fc1_forward_float32() raises:
    var r = create_fc1_params(DType.float32)
    LayerTester.test_linear_layer(400, 120, r[0], r[1], DType.float32)

fn test_fc1_forward_float16() raises:
    var r = create_fc1_params(DType.float16)
    LayerTester.test_linear_layer(400, 120, r[0], r[1], DType.float16)

fn test_fc1_backward_float32() raises:
    var r = create_fc1_params(DType.float32)
    LayerTester.test_linear_layer_backward(400, 120, r[0], r[1], DType.float32)

# FC2 tests
fn test_fc2_forward_float32() raises:
    var r = create_fc2_params(DType.float32)
    LayerTester.test_linear_layer(120, 84, r[0], r[1], DType.float32)

fn test_fc2_forward_float16() raises:
    var r = create_fc2_params(DType.float16)
    LayerTester.test_linear_layer(120, 84, r[0], r[1], DType.float16)

fn test_fc2_backward_float32() raises:
    var r = create_fc2_params(DType.float32)
    LayerTester.test_linear_layer_backward(120, 84, r[0], r[1], DType.float32)

# FC3 tests
fn test_fc3_forward_float32() raises:
    var r = create_fc3_params(DType.float32)
    LayerTester.test_linear_layer(84, 10, r[0], r[1], DType.float32)

fn test_fc3_forward_float16() raises:
    var r = create_fc3_params(DType.float16)
    LayerTester.test_linear_layer(84, 10, r[0], r[1], DType.float16)

fn test_fc3_backward_float32() raises:
    var r = create_fc3_params(DType.float32)
    LayerTester.test_linear_layer_backward(84, 10, r[0], r[1], DType.float32)

# Flatten tests
fn test_flatten_float32() raises:
    var input = create_special_value_tensor([1, 16, 5, 5], DType.float32, SPECIAL_VALUE_ONE)
    var flat = input.reshape([1, 400])
    assert_shape(flat, [1, 400], "Flatten shape")
    for i in range(flat.numel()):
        assert_false(isnan(flat._get_float64(i)), "NaN")

fn test_flatten_float16() raises:
    var input = create_special_value_tensor([1, 16, 5, 5], DType.float16, SPECIAL_VALUE_ONE)
    var flat = input.reshape([1, 400])
    assert_shape(flat, [1, 400], "Flatten shape")


fn main() raises:
    print("Running 24 tests to detect heap corruption...")

    var count = 0

    print("[1/24] conv1_forward_f32...", end="")
    test_conv1_forward_float32()
    print(" OK"); count += 1

    print("[2/24] conv1_forward_f16...", end="")
    test_conv1_forward_float16()
    print(" OK"); count += 1

    print("[3/24] conv1_backward_f32...", end="")
    test_conv1_backward_float32()
    print(" OK"); count += 1

    print("[4/24] conv2_forward_f32...", end="")
    test_conv2_forward_float32()
    print(" OK"); count += 1

    print("[5/24] conv2_forward_f16...", end="")
    test_conv2_forward_float16()
    print(" OK"); count += 1

    print("[6/24] conv2_backward_f32...", end="")
    test_conv2_backward_float32()
    print(" OK"); count += 1

    print("[7/24] relu_forward_f32...", end="")
    test_relu_forward_float32()
    print(" OK"); count += 1

    print("[8/24] relu_forward_f16...", end="")
    test_relu_forward_float16()
    print(" OK"); count += 1

    print("[9/24] relu_backward_f32...", end="")
    test_relu_backward_float32()
    print(" OK"); count += 1

    print("[10/24] maxpool1_forward_f32...", end="")
    test_maxpool1_forward_float32()
    print(" OK"); count += 1

    print("[11/24] maxpool1_forward_f16...", end="")
    test_maxpool1_forward_float16()
    print(" OK"); count += 1

    print("[12/24] maxpool2_forward_f32...", end="")
    test_maxpool2_forward_float32()
    print(" OK"); count += 1

    print("[13/24] maxpool2_forward_f16...", end="")
    test_maxpool2_forward_float16()
    print(" OK"); count += 1

    print("[14/24] fc1_forward_f32...", end="")
    test_fc1_forward_float32()
    print(" OK"); count += 1

    print("[15/24] fc1_forward_f16...", end="")
    test_fc1_forward_float16()
    print(" OK"); count += 1

    print("[16/24] fc1_backward_f32...", end="")
    test_fc1_backward_float32()
    print(" OK"); count += 1

    print("[17/24] fc2_forward_f32...", end="")
    test_fc2_forward_float32()
    print(" OK"); count += 1

    print("[18/24] fc2_forward_f16...", end="")
    test_fc2_forward_float16()
    print(" OK"); count += 1

    print("[19/24] fc2_backward_f32...", end="")
    test_fc2_backward_float32()
    print(" OK"); count += 1

    print("[20/24] fc3_forward_f32...", end="")
    test_fc3_forward_float32()
    print(" OK"); count += 1

    print("[21/24] fc3_forward_f16...", end="")
    test_fc3_forward_float16()
    print(" OK"); count += 1

    print("[22/24] fc3_backward_f32...", end="")
    test_fc3_backward_float32()
    print(" OK"); count += 1

    print("[23/24] flatten_f32...", end="")
    test_flatten_float32()
    print(" OK"); count += 1

    print("[24/24] flatten_f16...", end="")
    test_flatten_float16()
    print(" OK"); count += 1

    print("")
    print("ALL 24 TESTS PASSED - No heap corruption!")
'''


def main():
    os.chdir(REPO_ROOT)

    # Create temporary test file
    test_file = REPO_ROOT / "tests" / "models" / "_bisect_heap_test.mojo"

    try:
        # Write the test file
        test_file.write_text(TEST_FILE_CONTENT)

        # Try to run the test
        print(f"Running heap corruption test at commit {get_current_commit()}...")
        result = subprocess.run(
            ["pixi", "run", "mojo", "run", str(test_file)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=REPO_ROOT
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode == 0 and "ALL 24 TESTS PASSED" in result.stdout:
            print("\n✅ Bug is FIXED at this commit")
            return 0  # good commit - bug is fixed
        else:
            print(f"\n❌ Bug is PRESENT at this commit (exit code: {result.returncode})")
            return 1  # bad commit - bug is present

    except subprocess.TimeoutExpired:
        print("\n⚠️ Test timed out - treating as bug present")
        return 1
    except FileNotFoundError:
        print("\n⚠️ Cannot run test (mojo not found?) - skipping commit")
        return 125  # skip this commit
    except Exception as e:
        print(f"\n⚠️ Error: {e} - skipping commit")
        return 125
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()


def get_current_commit():
    """Get the current commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    return result.stdout.strip()


if __name__ == "__main__":
    sys.exit(main())
