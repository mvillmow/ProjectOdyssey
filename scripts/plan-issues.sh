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

# Function to parse reset time and wait until it expires
wait_for_rate_limit_reset() {
    local reset_time_str="$1"
    local timezone="$2"

    echo ""
    echo "=========================================="
    echo "  RATE LIMIT REACHED"
    echo "  Resets at: $reset_time_str ($timezone)"
    echo "=========================================="

    # Convert reset time to seconds since epoch
    # Handle formats like "2pm", "2:30pm", "14:00"
    local reset_time
    if [[ "$reset_time_str" =~ ^([0-9]{1,2}):?([0-9]{2})?(am|pm)?$ ]]; then
        local hour="${BASH_REMATCH[1]}"
        local min="${BASH_REMATCH[2]:-00}"
        local ampm="${BASH_REMATCH[3]:-}"

        # Convert to 24-hour format
        if [[ "$ampm" == "pm" && "$hour" -lt 12 ]]; then
            hour=$((hour + 12))
        elif [[ "$ampm" == "am" && "$hour" -eq 12 ]]; then
            hour=0
        fi

        # Get today's date and construct full datetime
        local today=$(TZ="$timezone" date +%Y-%m-%d)
        reset_time=$(TZ="$timezone" date -d "$today $hour:$min:00" +%s 2>/dev/null || echo "")

        # If reset time is in the past, it might be tomorrow
        local now=$(date +%s)
        if [[ -n "$reset_time" && "$reset_time" -lt "$now" ]]; then
            reset_time=$((reset_time + 86400))
        fi
    fi

    if [[ -z "$reset_time" ]]; then
        echo "  Could not parse reset time, waiting 60 minutes..."
        reset_time=$(($(date +%s) + 3600))
    fi

    # Countdown loop
    while true; do
        local now=$(date +%s)
        local remaining=$((reset_time - now))

        if [[ $remaining -le 0 ]]; then
            echo ""
            echo "  Rate limit reset! Resuming..."
            echo "=========================================="
            echo ""
            break
        fi

        local hours=$((remaining / 3600))
        local mins=$(((remaining % 3600) / 60))
        local secs=$((remaining % 60))

        printf "\r  Resuming in: %02d:%02d:%02d " $hours $mins $secs
        sleep 1
    done
}

# Function to check if plan output indicates rate limit
check_rate_limit() {
    local plan_file="$1"

    if grep -q "Limit reached" "$plan_file" 2>/dev/null; then
        # Extract reset time and timezone
        local limit_line=$(grep "Limit reached" "$plan_file" | head -1)
        if [[ "$limit_line" =~ resets[[:space:]]+([0-9]{1,2}:?[0-9]{0,2}[ap]?m?)[[:space:]]*\(([^)]+)\) ]]; then
            echo "${BASH_REMATCH[1]}|${BASH_REMATCH[2]}"
            return 0
        fi
        # Return generic indicator if we can't parse
        echo "unknown|America/Los_Angeles"
        return 0
    fi
    return 1
}

# Function to check if existing plan on GitHub has rate limit message
check_existing_plan_has_limit() {
    local issue_number="$1"

    local comments=$(gh issue view "$issue_number" --comments --json comments --jq '.comments[].body' 2>/dev/null || echo "")
    if echo "$comments" | grep -q "## Detailed Implementation Plan" && echo "$comments" | grep -q "Limit reached"; then
        return 0
    fi
    return 1
}

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
        # Check if existing plan was rate-limited (auto-replan these)
        if check_existing_plan_has_limit "$issue_number"; then
            echo "  Existing plan was rate-limited, will replan..."
        elif gh issue view "$issue_number" --comments --json comments --jq '.comments[].body' 2>/dev/null | grep -q "## Detailed Implementation Plan"; then
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
    # Always use default permission mode - 'plan' mode doesn't work with -p (print)
    # Control capabilities through --allowedTools instead
    if $AUTO_MODE; then
        ALLOWED_TOOLS="Read,Glob,Grep,WebFetch,WebSearch,Bash"
    else
        ALLOWED_TOOLS="Read,Glob,Grep,WebFetch,WebSearch"
    fi
    PERMISSION_MODE="default"

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
    # Retry loop for rate limiting
    while true; do
        echo "  Generating plan with Claude Opus..."
        echo "  (This may take 1-3 minutes. Check $log_file for progress)"
        start_time=$(date +%s)

        # Clear previous attempt
        > "$plan_file"
        > "$log_file"

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

        # Check for rate limit in output
        if limit_info=$(check_rate_limit "$plan_file"); then
            reset_time=$(echo "$limit_info" | cut -d'|' -f1)
            timezone=$(echo "$limit_info" | cut -d'|' -f2)
            wait_for_rate_limit_reset "$reset_time" "$timezone"
            echo "  Retrying plan generation..."
            continue
        fi

        # Also check stderr/log file for rate limit
        if limit_info=$(check_rate_limit "$log_file"); then
            reset_time=$(echo "$limit_info" | cut -d'|' -f1)
            timezone=$(echo "$limit_info" | cut -d'|' -f2)
            wait_for_rate_limit_reset "$reset_time" "$timezone"
            echo "  Retrying plan generation..."
            continue
        fi

        # No rate limit, break out of retry loop
        break
    done

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
