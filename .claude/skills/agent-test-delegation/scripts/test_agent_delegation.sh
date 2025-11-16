#!/usr/bin/env bash
#
# Test delegation for a specific agent
#
# Usage:
#   ./test_agent_delegation.sh <agent-name>

set -euo pipefail

AGENT_NAME="${1:-}"

if [[ -z "$AGENT_NAME" ]]; then
    echo "Error: Agent name required"
    echo "Usage: $0 <agent-name>"
    exit 1
fi

AGENT_FILE=".claude/agents/$AGENT_NAME.md"

if [[ ! -f "$AGENT_FILE" ]]; then
    echo "Error: Agent file not found: $AGENT_FILE"
    exit 1
fi

echo "Testing delegation for: $AGENT_NAME"
echo ""

# Extract frontmatter
FRONTMATTER=$(sed -n '/^---$/,/^---$/p' "$AGENT_FILE" | sed '1d;$d')

# Get level
LEVEL=$(echo "$FRONTMATTER" | grep "^level:" | cut -d':' -f2 | tr -d ' ')
echo "Level: $LEVEL"

# Get delegates_to
DELEGATES=$(echo "$FRONTMATTER" | grep "^delegates_to:" | cut -d':' -f2-)
if [[ -n "$DELEGATES" ]]; then
    echo ""
    echo "Delegates to:"
    echo "$DELEGATES" | tr -d '[]"' | tr ',' '\n' | while read -r delegate; do
        delegate=$(echo "$delegate" | tr -d ' ')
        if [[ -n "$delegate" ]]; then
            if [[ -f ".claude/agents/$delegate.md" ]]; then
                delegate_level=$(sed -n '/^---$/,/^---$/p' ".claude/agents/$delegate.md" | sed '1d;$d' | grep "^level:" | cut -d':' -f2 | tr -d ' ')
                echo "  ✅ $delegate (L$delegate_level)"

                # Check if delegation is to lower level
                if [[ "$delegate_level" -le "$LEVEL" ]]; then
                    echo "     ⚠️  Warning: Delegates to same or higher level"
                fi
            else
                echo "  ❌ $delegate (not found)"
            fi
        fi
    done
else
    echo "  (no delegation configured)"
fi

# Get escalates_to
ESCALATES=$(echo "$FRONTMATTER" | grep "^escalates_to:" | cut -d':' -f2-)
if [[ -n "$ESCALATES" ]]; then
    echo ""
    echo "Escalates to:"
    echo "$ESCALATES" | tr -d '[]"' | tr ',' '\n' | while read -r escalate; do
        escalate=$(echo "$escalate" | tr -d ' ')
        if [[ -n "$escalate" ]]; then
            if [[ -f ".claude/agents/$escalate.md" ]]; then
                escalate_level=$(sed -n '/^---$/,/^---$/p' ".claude/agents/$escalate.md" | sed '1d;$d' | grep "^level:" | cut -d':' -f2 | tr -d ' ')
                echo "  ✅ $escalate (L$escalate_level)"

                # Check if escalation is to higher level
                if [[ "$escalate_level" -ge "$LEVEL" ]]; then
                    echo "     ⚠️  Warning: Escalates to same or lower level"
                fi
            else
                echo "  ❌ $escalate (not found)"
            fi
        fi
    done
else
    echo ""
    echo "Escalates to:"
    echo "  (no escalation configured)"
fi

echo ""
