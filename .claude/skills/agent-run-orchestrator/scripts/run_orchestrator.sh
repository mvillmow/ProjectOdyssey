#!/usr/bin/env bash
#
# Run section orchestrator
#
# Usage:
#   ./run_orchestrator.sh <section-name>

set -euo pipefail

SECTION="${1:-}"

if [[ -z "$SECTION" ]]; then
    echo "Error: Section name required"
    echo "Usage: $0 <section-name>"
    echo ""
    echo "Available sections:"
    echo "  foundation"
    echo "  shared-library"
    echo "  tooling"
    echo "  first-paper"
    echo "  ci-cd"
    echo "  agentic-workflows"
    exit 1
fi

# Map section to orchestrator
ORCHESTRATOR="${SECTION}-orchestrator"

echo "Running $ORCHESTRATOR..."
echo ""

# Check if orchestrator exists
AGENT_FILE=".claude/agents/$ORCHESTRATOR.md"
if [[ ! -f "$AGENT_FILE" ]]; then
    echo "Error: Orchestrator not found: $AGENT_FILE"
    exit 1
fi

# Display orchestrator info
echo "Orchestrator: $ORCHESTRATOR"
echo "Configuration: $AGENT_FILE"
echo ""

# Show section plan
PLAN_FILE="notes/plan/$(echo $SECTION | sed 's/-/\//g')/plan.md"
if [[ -f "$PLAN_FILE" ]]; then
    echo "Section Plan: $PLAN_FILE"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    head -20 "$PLAN_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "Warning: Plan not found: $PLAN_FILE"
fi

echo ""
echo "Orchestrator loaded. Ready to coordinate section work."
echo ""
echo "Typical workflow:"
echo "1. Review section plan"
echo "2. Break into components"
echo "3. Delegate to design agents"
echo "4. Monitor progress"
echo "5. Integrate results"
