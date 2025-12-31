#!/bin/bash
# Run git bisect to find the commit that fixed heap corruption bug #2942
#
# Usage:
#   ./scripts/run_bisect_heap.sh
#
# This will:
# 1. Copy the test script to /tmp (so it persists across checkouts)
# 2. Start git bisect between HEAD (good) and the split commit (bad)
# 3. Run the test script at each commit
# 4. Report the first commit that fixed the bug

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

echo "=============================================="
echo "Git Bisect: Finding Heap Corruption Fix"
echo "=============================================="
echo ""
echo "Good commit (bug fixed): HEAD"
echo "Bad commit (bug present): aee64aea~1 (before split workaround)"
echo ""

# Copy the test script to /tmp so it persists across git checkouts
cp "$SCRIPT_DIR/bisect_heap_test.py" /tmp/bisect_heap_test.py
echo "Copied test script to /tmp/bisect_heap_test.py"
echo ""

# Clean up any existing bisect
git bisect reset 2>/dev/null || true

# Start bisect
# HEAD = good (tests pass now)
# aee64aea~1 = bad (this is where the bug was present)
git bisect start HEAD aee64aea~1

# Run the bisect with our test script from /tmp
echo "Running bisect..."
echo ""

git bisect run python3 /tmp/bisect_heap_test.py

echo ""
echo "=============================================="
echo "Bisect Complete"
echo "=============================================="

# Show the result
git bisect log | tail -20

# Clean up
rm -f /tmp/bisect_heap_test.py

# Reset
git bisect reset

echo ""
echo "Done! The first good commit above is the one that fixed the bug."
