#!/bin/bash
# plan-issues.sh - Generate and post implementation plans for GitHub issues
#
# Usage: plan-issues.sh [OPTIONS]
#   --limit N        Only process first N issues (default: all)
#   --auto           Non-interactive mode: skip vim, auto-post plans, allow gh CLI
#   --replan         Re-plan issues that already have a plan comment
#   --issues N,M,... Only process specific issue numbers (comma-separated)
#
set -euo pipefail

# Parse arguments
LIMIT=""
AUTO_MODE=false
REPLAN_MODE=false
REPLAN_REASON=""
SPECIFIC_ISSUES=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --auto)
            AUTO_MODE=true
            shift
            ;;
        --replan)
            REPLAN_MODE=true
            shift
            ;;
        --replan-reason)
            REPLAN_MODE=true
            REPLAN_REASON="$2"
            shift 2
            ;;
        --issues)
            SPECIFIC_ISSUES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: plan-issues.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --limit N            Only process first N issues (default: all)"
            echo "  --auto               Non-interactive mode: skip vim, auto-post plans, allow gh CLI"
            echo "  --replan             Re-plan issues that already have a plan comment"
            echo "  --replan-reason TXT  Re-plan with additional context (implies --replan)"
            echo "  --issues N,M,...     Only process specific issue numbers (comma-separated)"
            echo ""
            echo "Examples:"
            echo "  plan-issues.sh --limit 5                    # First 5 open issues"
            echo "  plan-issues.sh --issues 123,456,789         # Specific issues only"
            echo "  plan-issues.sh --auto --replan              # Auto mode, allow replanning"
            echo "  plan-issues.sh --issues 123 --replan-reason 'Need to add error handling'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHIEF_ARCHITECT_PROMPT=$(cat "$REPO_ROOT/.claude/agents/chief-architect.md")

# Create temp directory for this run
RUN_ID=$(date +%Y%m%d_%H%M%S)
TEMP_DIR="/tmp/plan-issues-${RUN_ID}"
mkdir -p "$TEMP_DIR"
echo "Temp directory: $TEMP_DIR"
if $AUTO_MODE; then
    echo "Mode: AUTO (non-interactive, plans auto-posted)"
else
    echo "Mode: INTERACTIVE (vim review before posting)"
fi
if $REPLAN_MODE; then
    echo "Replan: ENABLED (will re-plan issues with existing plans)"
    if [[ -n "$REPLAN_REASON" ]]; then
        echo "Replan reason: $REPLAN_REASON"
    fi
else
    echo "Replan: DISABLED (will skip issues with existing plans)"
fi

# Get issue numbers based on options
if [[ -n "$SPECIFIC_ISSUES" ]]; then
    # Use specific issues provided by user
    IFS=',' read -ra issues <<< "$SPECIFIC_ISSUES"
    echo "Issues: Using specific issues: ${issues[*]}"
else
    # Get all open issue numbers into array
    mapfile -t all_issues < <(gh issue list --state open --limit 500 --json number --jq '.[].number' | sort -n)

    # Apply limit if specified
    if [[ -n "$LIMIT" ]]; then
        issues=("${all_issues[@]:0:$LIMIT}")
    else
        issues=("${all_issues[@]}")
    fi
fi
total=${#issues[@]}
current=0
skipped=0
posted=0

echo "=========================================="
echo "  Issue Planning Script"
echo "  Total open issues: $total"
echo "=========================================="
echo ""

for issue_number in "${issues[@]}"; do
    current=$((current + 1))

    # Get issue details
    issue_title=$(gh issue view "$issue_number" --json title --jq '.title')
    issue_body=$(gh issue view "$issue_number" --json body --jq '.body')

    echo "[$current/$total] Issue #${issue_number}: ${issue_title}"
    echo "----------------------------------------"

    # Check if issue already has a plan (unless in replan mode)
    if ! $REPLAN_MODE; then
        if gh issue view "$issue_number" --comments --json comments --jq '.comments[].body' 2>/dev/null | grep -q "## Detailed Implementation Plan"; then
            echo "  SKIPPED (already has plan - use --replan to override)"
            skipped=$((skipped + 1))
            echo ""
            continue
        fi
    fi

    # Create files in temp directory
    plan_file="${TEMP_DIR}/issue-${issue_number}-plan.md"
    log_file="${TEMP_DIR}/issue-${issue_number}-claude.log"
    cmd_file="${TEMP_DIR}/issue-${issue_number}-command.sh"

    echo "  Plan file: $plan_file"
    echo "  Log file:  $log_file"

    # Build the prompt
    REPLAN_CONTEXT=""
    if $REPLAN_MODE; then
        REPLAN_CONTEXT="

NOTE: This is a REPLAN request. A previous plan exists for this issue."
        if [[ -n "$REPLAN_REASON" ]]; then
            REPLAN_CONTEXT="${REPLAN_CONTEXT}
REPLAN REASON: ${REPLAN_REASON}

Please review the existing plan comments on this issue and create an updated plan that addresses the replan reason."
        else
            REPLAN_CONTEXT="${REPLAN_CONTEXT}

Please review the existing plan comments on this issue and create an improved/updated plan."
        fi
    fi

    PROMPT="Create a detailed implementation plan for the following GitHub issue:

Issue #${issue_number}: ${issue_title}

${issue_body}
${REPLAN_CONTEXT}

BUDGET: You have a maximum of 50 tool calls and 50 steps. Use them wisely to explore the codebase and create a thorough plan.

Output a comprehensive plan in markdown format with:
1. Summary of what needs to be done
2. Step-by-step implementation tasks
3. Files to modify/create
4. Testing approach
5. Success criteria

At the end of your response, include:
## Resource Usage
- Tool calls used: X/50
- Steps used: Y/50"

    # Set tools based on mode
    if $AUTO_MODE; then
        ALLOWED_TOOLS="Read,Glob,Grep,WebFetch,WebSearch,Bash"
        PERMISSION_MODE="default"
    else
        ALLOWED_TOOLS="Read,Glob,Grep,WebFetch,WebSearch"
        PERMISSION_MODE="plan"
    fi

    # Save command for debugging
    cat > "$cmd_file" << CMDEOF
claude --model opus \\
       --permission-mode $PERMISSION_MODE \\
       --allowedTools "$ALLOWED_TOOLS" \\
       --add-dir "$REPO_ROOT" \\
       --system-prompt "\$CHIEF_ARCHITECT_PROMPT" \\
       -p \\
       "\$PROMPT"
CMDEOF

    # Generate plan using Claude with opus and chief architect prompt
    echo "  Generating plan with Claude Opus..."
    echo "  (This may take 1-3 minutes. Check $log_file for progress)"
    start_time=$(date +%s)

    # Run Claude - tools and permissions vary by mode
    # stdout -> plan file, stderr -> log file
    claude --model opus \
           --permission-mode "$PERMISSION_MODE" \
           --allowedTools "$ALLOWED_TOOLS" \
           --add-dir "$REPO_ROOT" \
           --system-prompt "$CHIEF_ARCHITECT_PROMPT" \
           -p \
           "$PROMPT" > "$plan_file" 2> "$log_file" &

    CLAUDE_PID=$!

    # Progress indicator while waiting
    spin='-\|/'
    i=0
    while kill -0 $CLAUDE_PID 2>/dev/null; do
        i=$(( (i+1) % 4 ))
        printf "\r  Generating... ${spin:$i:1} (elapsed: $(($(date +%s) - start_time))s) "
        sleep 0.5
    done
    wait $CLAUDE_PID || true
    printf "\r                                              \r"

    end_time=$(date +%s)
    echo "  Generation completed in $((end_time - start_time))s"
    echo "  Plan size: $(wc -c < "$plan_file") bytes"

    # Interactive mode: open vim for review
    if ! $AUTO_MODE; then
        echo "  Opening vim for review... (delete all content to skip)"
        vim "$plan_file"

        # Check if file is empty (user wants to skip)
        if [[ ! -s "$plan_file" ]]; then
            echo "  SKIPPED (empty file)"
            skipped=$((skipped + 1))
            echo ""
            continue
        fi
    fi

    # Check if plan was generated
    if [[ ! -s "$plan_file" ]]; then
        echo "  ERROR: No plan generated (empty file)"
        skipped=$((skipped + 1))
        echo ""
        continue
    fi

    # Post plan to GitHub issue
    echo "  Posting plan to GitHub..."
    if $REPLAN_MODE && [[ -n "$REPLAN_REASON" ]]; then
        PLAN_HEADER="## Detailed Implementation Plan (Revised)

**Replan reason:** ${REPLAN_REASON}
"
    elif $REPLAN_MODE; then
        PLAN_HEADER="## Detailed Implementation Plan (Revised)
"
    else
        PLAN_HEADER="## Detailed Implementation Plan
"
    fi

    gh issue comment "$issue_number" --body "${PLAN_HEADER}
$(cat "$plan_file")"

    posted=$((posted + 1))
    echo "  Plan posted successfully!"
    echo ""
done

echo "=========================================="
echo "  Summary"
echo "  Posted: $posted"
echo "  Skipped: $skipped"
echo "  Total: $total"
echo ""
echo "  Temp directory preserved: $TEMP_DIR"
echo "  - Plans: issue-*-plan.md"
echo "  - Logs:  issue-*-claude.log"
echo "  - Cmds:  issue-*-command.sh"
echo "=========================================="
