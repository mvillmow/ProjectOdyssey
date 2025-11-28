#!/bin/bash
# Fix for issue #2128: YAML parsing for values containing colons

set -e

cd /home/mvillmow/ml-odyssey

# Create and switch to feature branch
git checkout -b 2128-fix-yaml-colon-parsing

# Stage the changes
git add shared/utils/config.mojo
git add notes/issues/2128/README.md

# Commit with message
git commit -m "fix(config): Handle colons in JSON values

- Fixed JSON parser to only split on first colon, not all colons
- Handles URLs like http://example.com, timestamps, port numbers
- Changed from split() to find() + slicing for proper parsing
- Resolves issue #2128

Closes #2128"

# Display status
echo ""
echo "✓ Branch created: 2128-fix-yaml-colon-parsing"
echo "✓ Changes committed"
echo ""
echo "Next step: git push -u origin 2128-fix-yaml-colon-parsing"
