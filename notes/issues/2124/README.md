# Issue #2124: Fix test_validate_model_config test failure

## Objective

Fix failing test in test_validation.mojo that expects incorrect configuration keys.

## Problem

The `test_validate_model_config()` test was checking for keys "name" and "num_classes" at the root level, but the LeNet-5 model configuration file uses a nested structure with these keys under "model":

- Expected: `name` → Actual: `model.name`
- Expected: `num_classes` → Actual: `model.output_classes`

The test also incorrectly attempted to validate `num_classes` instead of `model.output_classes`.

## Root Cause

The test had incorrect assumptions about key naming and structure. When configuration files are loaded via `load_config()`, nested YAML structures are flattened into dot-notation keys (e.g., "model.name", "model.output_classes").

Looking at the same test file, other tests correctly use dot notation:
- Line 373-376: `"optimizer.name"`, `"optimizer.learning_rate"`
- Line 374-377: `"training.epochs"`, `"training.batch_size"`

## Solution

Updated `test_validate_model_config()` to use correct dot-notation keys:

1. Changed `"name"` to `"model.name"`
2. Changed `"num_classes"` to `"model.output_classes"`
3. Updated range validation to use `"model.output_classes"`

## Changes Made

File: `/home/mvillmow/ml-odyssey/tests/configs/test_validation.mojo`

- Lines 433-434: Updated required keys list
- Line 439: Updated range validation key

## Verification

The fix aligns test expectations with:
1. Actual structure of `configs/papers/lenet5/model.yaml`
2. Configuration loading behavior (nested dict flattening)
3. Pattern used in other tests in the same file

## Files Modified

- `tests/configs/test_validation.mojo` - Fixed test expectations

## Success Criteria

- [x] Test expects correct dot-notation keys matching actual config file structure
- [x] Test references actual config field names (model.output_classes not num_classes)
- [x] Test pattern now matches other config validation tests in same file
