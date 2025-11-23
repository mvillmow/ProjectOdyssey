# Test Failures - FIXME Summary

**Total Tests**: 146
**Passed**: 140 (96%)
**Failed**: 6 (4%)

---

## Quick Fix Checklist

### 1. Configuration Schema Issues (3 failures)

Location: `tests/configs/test_schema.py`

Tests Failing:

- `test_lenet5_model_config_validates`
- `test_all_default_configs_validate`
- `test_all_lenet5_configs_validate`

**Root Cause**: Model configuration format mismatch (uses `architecture` dict, schema expects `layers` array)

**FIXME**:

```
FIXME: /home/mvillmow/ml-odyssey/configs/schemas/model.schema.yaml
  Update schema or decide on format standard

FIXME: /home/mvillmow/ml-odyssey/configs/papers/lenet5/model.yaml
  Update to match schema format

FIXME: /home/mvillmow/ml-odyssey/configs/defaults/model.yaml
  Ensure compliance with schema
```

**Effort**: 1 hour

---

### 2. Batch Size Threshold Not Implemented (1 failure)

Location: `tests/scripts/test_lint_configs.py::test_batch_size_too_large`

**Root Cause**: Performance threshold checking logic missing in linter

**FIXME**:

```
FIXME: /home/mvillmow/ml-odyssey/scripts/lint_configs.py
  In _check_performance_thresholds() or similar method:
  - Add warning if batch_size > 2048
  - Add warning if batch_size < 8

  Expected behavior:
    batch_size: 10000  →  warning: "Batch size very large"
```

**Code Template**:

```python
def _check_batch_size(self, batch_size):
    if batch_size and batch_size > 2048:
        self.suggestions.append(
            f"Batch size {batch_size} is very large (typical: 8-2048)"
        )
```

**Effort**: 30 minutes

---

### 3. Correct Indentation Test Failure (1 failure)

Location: `tests/scripts/test_lint_configs.py::test_correct_indentation`

**Root Cause**: Test YAML missing required `training.batch_size` key

**FIXME**:

```
FIXME: /home/mvillmow/ml-odyssey/tests/scripts/test_lint_configs.py
  Line 77-82: Update test YAML to include all required keys

  Current:
    training:
      epochs: 10
      optimizer:
        name: adam
        learning_rate: 0.001

  Fixed:
    training:
      epochs: 10
      batch_size: 32  # ← ADD THIS
      optimizer:
        name: adam
        learning_rate: 0.001
```

**Effort**: 5 minutes

---

### 4. Section Detection Logic (1 failure)

Location: `tests/test_validation.py::test_check_all_sections_present`

**Root Cause**: Pattern matching not correctly extracting markdown section names

**FIXME**:

```
FIXME: /home/mvillmow/ml-odyssey/scripts/validation.py
  In check_required_sections() function:

  Current Issue:
    - Not detecting "## Section 1" as containing "Section 1"
    - Missing sections list returned when should be found

  Fix Strategy:
    1. Use regex: r'^#{1,6}\s+(.+?)$' to extract heading text
    2. Strip whitespace: h.strip()
    3. Compare case-insensitively: section.lower() in h.lower()

  Test Case:
    Content: "## Section 1\nContent here."
    Required: ["Section 1", "Section 2"]
    Expected missing: ["Section 2"]
    Current missing: ["Section 1", "Section 2"]
```

**Code Template**:

```python
import re

def check_required_sections(content, required_sections):
    """Check if all required sections are present."""
    # Extract heading text after ##
    headings = re.findall(r'^#{1,6}\s+(.+?)$', content, re.MULTILINE)
    heading_texts = [h.strip() for h in headings]

    missing = []
    for section in required_sections:
        # Case-insensitive search
        if not any(section.lower() in h.lower() for h in heading_texts):
            missing.append(section)

    return len(missing) == 0, missing
```

**Effort**: 30 minutes

---

## Priority Matrix

| Fix | Severity | Effort | Priority |
|-----|----------|--------|----------|
| Schema mismatch | HIGH | 1hr | 🔴 DO FIRST |
| Batch size threshold | MEDIUM | 30min | 🟡 DO SECOND |
| Test YAML fix | LOW | 5min | 🟢 QUICK FIX |
| Section detection | LOW | 30min | 🟢 EASY |

**Total Effort**: 2-2.5 hours

---

## Test Execution Command

```bash
# Run all tests
cd /home/mvillmow/ml-odyssey
pytest tests/ -v --tb=short

# Run specific failing tests
pytest tests/configs/test_schema.py::test_lenet5_model_config_validates -v
pytest tests/scripts/test_lint_configs.py::test_batch_size_too_large -v
pytest tests/scripts/test_lint_configs.py::test_correct_indentation -v
pytest tests/test_validation.py::test_check_all_sections_present -v

# Run by category
pytest tests/configs/ -v      # Schema tests
pytest tests/scripts/ -v      # Linting tests
pytest tests/test_validation.py -v  # Validation tests
```

---

## Success Criteria

Target: 100% pass rate (146/146 tests)

Current: 140/146 (96%)

**After fixes**: 146/146 (100%) ✅
