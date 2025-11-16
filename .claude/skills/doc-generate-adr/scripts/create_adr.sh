#!/usr/bin/env bash
#
# Create a new Architecture Decision Record (ADR)
#
# Usage:
#   ./create_adr.sh "Decision Title"

set -euo pipefail

TITLE="${1:-}"

if [[ -z "$TITLE" ]]; then
    echo "Error: ADR title required"
    echo "Usage: $0 \"Decision Title\""
    exit 1
fi

ADR_DIR="notes/review/adr"
mkdir -p "$ADR_DIR"

# Find next ADR number
LAST_NUMBER=0
for file in "$ADR_DIR"/ADR-*.md; do
    if [[ -f "$file" ]]; then
        NUM=$(basename "$file" | sed 's/ADR-\([0-9]*\)-.*/\1/')
        if [[ "$NUM" =~ ^[0-9]+$ ]] && [[ $NUM -gt $LAST_NUMBER ]]; then
            LAST_NUMBER=$NUM
        fi
    fi
done

NEXT_NUMBER=$((LAST_NUMBER + 1))
ADR_NUMBER=$(printf "%03d" $NEXT_NUMBER)

# Create filename
FILENAME=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | sed 's/[^a-z0-9-]//g')
ADR_FILE="$ADR_DIR/ADR-$ADR_NUMBER-$FILENAME.md"

# Create ADR from template
cat > "$ADR_FILE" <<EOF
# ADR-$ADR_NUMBER: $TITLE

**Status**: Proposed

**Date**: $(date +%Y-%m-%d)

**Deciders**: [Names or roles]

## Context

What is the issue we're facing? What factors are we considering?

## Decision

What is the decision we're making?

## Rationale

Why are we making this decision? What are the key reasons?

- Reason 1
- Reason 2
- Reason 3

## Consequences

### Positive

- Benefit 1
- Benefit 2

### Negative

- Drawback 1
- Drawback 2

### Neutral

- Other impact 1

## Alternatives Considered

### Alternative 1

Description of alternative and why it was not chosen.

### Alternative 2

Description of alternative and why it was not chosen.

## References

- [Related documentation]
- [Evidence or research]
- [Related ADRs]
EOF

echo "✅ ADR created: $ADR_FILE"
echo ""
echo "Next steps:"
echo "1. Edit the ADR to fill in details"
echo "2. Review with team"
echo "3. Update status to 'Accepted' when approved"
echo "4. Commit to repository"
