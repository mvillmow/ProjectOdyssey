#!/usr/bin/env bash
#
# Validate a single agent configuration file
#
# Usage:
#   ./validate_agent.sh <agent-file>

set -euo pipefail

AGENT_FILE="${1:-}"

if [[ -z "$AGENT_FILE" ]] || [[ ! -f "$AGENT_FILE" ]]; then
    echo "Error: Valid agent file required"
    echo "Usage: $0 <agent-file>"
    exit 1
fi

echo "Validating: $AGENT_FILE"
echo ""

ERRORS=0

# Extract YAML frontmatter
FRONTMATTER=$(sed -n '/^---$/,/^---$/p' "$AGENT_FILE" | sed '1d;$d')

if [[ -z "$FRONTMATTER" ]]; then
    echo "❌ No YAML frontmatter found"
    ((ERRORS++))
    exit 1
fi

echo "✅ YAML frontmatter found"

# Check required fields
REQUIRED_FIELDS=("name" "role" "level" "phase" "description")

for field in "${REQUIRED_FIELDS[@]}"; do
    if echo "$FRONTMATTER" | grep -q "^$field:"; then
        echo "✅ Required field: $field"
    else
        echo "❌ Missing required field: $field"
        ((ERRORS++))
    fi
done

# Validate level (must be 0-5)
LEVEL=$(echo "$FRONTMATTER" | grep "^level:" | cut -d':' -f2 | tr -d ' ')
if [[ -n "$LEVEL" ]]; then
    if [[ "$LEVEL" =~ ^[0-5]$ ]]; then
        echo "✅ Valid level: $LEVEL"
    else
        echo "❌ Invalid level: $LEVEL (must be 0-5)"
        ((ERRORS++))
    fi
fi

# Validate phase
PHASE=$(echo "$FRONTMATTER" | grep "^phase:" | cut -d':' -f2 | tr -d ' ')
if [[ -n "$PHASE" ]]; then
    VALID_PHASES=("Plan" "Test" "Implementation" "Package" "Cleanup")
    if [[ " ${VALID_PHASES[@]} " =~ " ${PHASE} " ]]; then
        echo "✅ Valid phase: $PHASE"
    else
        echo "❌ Invalid phase: $PHASE"
        echo "   Valid phases: ${VALID_PHASES[*]}"
        ((ERRORS++))
    fi
fi

# Validate tools (if present)
if echo "$FRONTMATTER" | grep -q "^tools:"; then
    VALID_TOOLS=("Read" "Write" "Bash" "Grep" "Glob")
    TOOLS_LINE=$(echo "$FRONTMATTER" | grep "^tools:" | cut -d':' -f2-)

    # Simple validation (just check format)
    if [[ "$TOOLS_LINE" =~ \[.*\] ]]; then
        echo "✅ Tools field properly formatted"
    else
        echo "⚠️  Tools field may be improperly formatted"
    fi
fi

echo ""
if [[ $ERRORS -eq 0 ]]; then
    echo "✅ Validation passed"
    exit 0
else
    echo "❌ Validation failed with $ERRORS error(s)"
    exit 1
fi
