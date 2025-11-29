# Foundation Tests - Quick Fix Guide

## Status

- **Tests Passing**: 154/156 (98.7%)
- **Tests Failing**: 2
- **Tests Skipped**: 10 (intentional, not failures)

## The Problem

Two tests fail because there are extra directories in `/docs/`:

```text
/home/mvillmow/ml-odyssey/docs/
├── getting-started/ .................. ✓ Expected
├── core/ ............................ ✓ Expected
├── advanced/ ....................... ✓ Expected
├── dev/ ............................. ✓ Expected
├── integration/ ..................... ✓ Expected
├── backward-passes/ ................. ✗ UNEXPECTED
└── extensor/ ........................ ✗ UNEXPECTED
```text

### Failing Tests

1. `test_doc_structure.py::TestDocumentationStructure::test_no_unexpected_directories`
2. `test_doc_structure.py::TestDocumentationHierarchy::test_tier_count`

## The Fix

### Choose One Option

#### Option A: Delete (Recommended if not needed)

```bash
rm -rf /home/mvillmow/ml-odyssey/docs/backward-passes/
rm -rf /home/mvillmow/ml-odyssey/docs/extensor/
```text

**Note**: `backward-passes/` has restricted permissions (700). Check contents before deleting.

#### Option B: Reorganize (If content should be preserved)

```bash
# Move content to appropriate tier
# Examples:
# - backward-passes/ → advanced/gradient-computation.md
# - extensor/ → advanced/tensor-extensions.md
# - or move to core/ if foundational
```text

## Verify the Fix

```bash
cd /home/mvillmow/ml-odyssey
pytest tests/foundation/docs/test_doc_structure.py -v
```text

Expected result:

```text
===== 24 passed in 0.06s =====
(14 passed from structure tests, 10 skipped from tier completion tests)
```text

Or run all foundation tests:

```bash
pytest tests/foundation/ -v --tb=short
```text

Expected result:

```text
===== 156 passed, 10 skipped in ~0.5s =====
```text

## Files to Modify

Only these 2 items need action:

- [ ] `/home/mvillmow/ml-odyssey/docs/backward-passes/` - DELETE or MOVE
- [ ] `/home/mvillmow/ml-odyssey/docs/extensor/` - DELETE or MOVE

## Optional: Enable Skipped Tests

To enable 5 skipped tests in tier 1, create:

```bash
touch /home/mvillmow/ml-odyssey/docs/getting-started/first-paper.md
```text

Add content:

```markdown
# Getting Started with Your First Paper

## LeNet-5: A Classic CNN Implementation

[Add tutorial content]
```text

This will enable:

- test_first_paper_exists
- test_first_paper_has_title
- test_first_paper_has_tutorial
- test_all_tier1_docs_exist
- test_tier1_document_count

But this is OPTIONAL - can be done later when implementing the first paper.

## Summary

| Item | Time | Priority |
|------|------|----------|
| Delete/move 2 directories | 5 min | ⚠️ REQUIRED |
| Re-run tests to verify | 1 min | ⚠️ REQUIRED |
| Create first-paper.md | 15 min | ℹ️ OPTIONAL |

**Total time to fix**: ~5-10 minutes
