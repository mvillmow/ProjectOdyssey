#!/bin/bash
# Script to apply and commit fix for issue #2128: YAML parsing for values containing colons

set -e

cd /home/mvillmow/ml-odyssey

echo "================================================================================"
echo "Applying fix for issue #2128: YAML Parsing for Values Containing Colons"
echo "================================================================================"
echo ""

# Check if we're on main branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "Not on main branch. Current branch: $CURRENT_BRANCH"
    echo "Switching to main..."
    git checkout main
fi

# Create feature branch
echo "Creating feature branch: 2128-fix-yaml-colon-parsing"
git checkout -b 2128-fix-yaml-colon-parsing || git checkout 2128-fix-yaml-colon-parsing

# Stage all changes
echo "Staging changes..."
git add -A

# Verify what will be committed
echo ""
echo "Changes to be committed:"
git diff --cached --stat

echo ""
echo "Creating commit..."
git commit -m "fix(config): Handle colons in JSON values

JSON parser now only splits on the first colon, correctly handling
values that contain colons such as URLs, timestamps, and port numbers.

Changes:
- Replaced pair.split(':') with find()+slicing approach
- Fixed parsing of http://example.com, postgresql://user:pass@host
- Handles timestamps like 12:30:45 and multi-colon values

Tests:
- Added tests/configs/fixtures/urls.json test fixture
- Added tests/configs/test_json_colon_values.mojo test suite
- Tests verify URL preservation, database URLs, timestamps

Closes #2128"

echo ""
echo "================================================================================"
echo "✓ Fix applied successfully!"
echo "================================================================================"
echo ""
echo "Branch: 2128-fix-yaml-colon-parsing"
echo "Commit: $(git rev-parse --short HEAD)"
echo ""
echo "Next steps:"
echo "  1. Verify: git show --stat"
echo "  2. Push:   git push -u origin 2128-fix-yaml-colon-parsing"
echo "  3. Create PR: gh pr create --title 'Fix JSON colon parsing' --body 'Closes #2128'"
echo ""
