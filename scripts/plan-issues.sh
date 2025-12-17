#!/bin/bash
# plan-issues.sh - Generate and post implementation plans for GitHub issues
#
# Usage: plan-issues.sh [OPTIONS]
#   --limit N        Only process first N issues (default: all)
#   --auto           Non-interactive mode: skip vim, auto-post plans, allow gh CLI
#   --replan         Re-plan issues that already have a plan comment
#   --replan-reason  Re-plan with additional context (implies --replan)
#   --issues N,M,... Only process specific issue numbers (comma-separated)
#   --dry-run        Preview what will be done without posting to GitHub
#
set -euo pipefail

# Bash version check - mapfile requires bash 4.0+
if [[ "${BASH_VERSINFO[0]}" -lt 4 ]]; then
    echo "ERROR: This script requires bash 4.0+ (found ${BASH_VERSION})"
    echo "On macOS, install with: brew install bash"
    exit 1
fi

# Constants
readonly MAX_ISSUES_FETCH=500
readonly MAX_RETRIES=3
readonly CLAUDE_MAX_TOOLS=50
readonly CLAUDE_MAX_STEPS=50
readonly CLAUDE_TIMEOUT=${CLAUDE_TIMEOUT:-600}  # 10 minute timeout
readonly MAX_PARALLEL=${MAX_PARALLEL:-4}  # Parallel job limit
readonly ALLOWED_TIMEZONES="America/Los_Angeles|America/New_York|America/Chicago|America/Denver|America/Phoenix|UTC|Europe/London|Europe/Paris|Asia/Tokyo"
readonly APPROVED_EDITORS="vim|vi|emacs|nano|code|subl|nvim|helix|micro|edit"  # Whitelist for editors

# Structured logging functions
readonly LOG_PREFIX_INFO="[INFO]"
readonly LOG_PREFIX_WARN="[WARN]"
readonly LOG_PREFIX_ERROR="[ERROR]"
readonly LOG_PREFIX_DEBUG="[DEBUG]"

log_info() { echo "$LOG_PREFIX_INFO $(date '+%H:%M:%S') $*"; }
log_warn() { echo "$LOG_PREFIX_WARN $(date '+%H:%M:%S') $*" >&2; }
log_error() { echo "$LOG_PREFIX_ERROR $(date '+%H:%M:%S') $*" >&2; }
log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo "$LOG_PREFIX_DEBUG $(date '+%H:%M:%S') $*" >&2
    fi
}

# Function to parse reset time and wait until it expires
wait_for_rate_limit_reset() {
    local reset_time_str="$1"
    local timezone="$2"

    # Validate timezone against whitelist
    if ! echo "$timezone" | grep -qE "^($ALLOWED_TIMEZONES)$"; then
        echo "  WARNING: Invalid timezone '$timezone', defaulting to America/Los_Angeles"
        timezone="America/Los_Angeles"
    fi

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
        # Use portable date handling (GNU vs BSD/macOS)
        local today=$(TZ="$timezone" date +%Y-%m-%d)
        if date --version >/dev/null 2>&1; then
            # GNU date
            reset_time=$(TZ="$timezone" date -d "$today $hour:$min:00" +%s 2>/dev/null || echo "")
        else
            # BSD/macOS date
            reset_time=$(TZ="$timezone" date -j -f "%Y-%m-%d %H:%M:%S" "$today $hour:$min:00" +%s 2>/dev/null || echo "")
        fi

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

    # Allow user to interrupt the wait with Ctrl+C
    local interrupted=false
    trap 'echo ""; echo "  Wait interrupted by user"; interrupted=true' INT

    # Countdown loop
    while true; do
        if $interrupted; then
            trap - INT  # Restore default handler
            return 1
        fi

        local now=$(date +%s)
        local remaining=$((reset_time - now))

        if [[ $remaining -le 0 ]]; then
            echo ""
            echo "  Rate limit reset! Resuming..."
            echo "=========================================="
            echo ""
            trap - INT  # Restore default handler
            break
        fi

        local hours=$((remaining / 3600))
        local mins=$(((remaining % 3600) / 60))
        local secs=$((remaining % 60))

        printf "\r  Resuming in: %02d:%02d:%02d " $hours $mins $secs
        sleep 1
    done

    trap - INT  # Restore default handler
}

# Function to check if plan output indicates rate limit
check_rate_limit() {
    local plan_file="$1"

    if grep -q "Limit reached" "$plan_file" 2>/dev/null; then
        # Extract reset time and timezone using sed
        local limit_line=$(grep "Limit reached" "$plan_file" | head -1)
        # Parse "resets 2pm (America/Los_Angeles)" pattern
        local reset_time=$(echo "$limit_line" | sed -n 's/.*resets[[:space:]]*\([0-9:apm]*\)[[:space:]]*(.*/\1/p')
        local timezone=$(echo "$limit_line" | sed -n 's/.*(\([^)]*\)).*/\1/p')

        if [[ -n "$reset_time" && -n "$timezone" ]]; then
            echo "${reset_time}|${timezone}"
            return 0
        fi
        # Return generic indicator if we can't parse
        echo "unknown|America/Los_Angeles"
        return 0
    fi
    return 1
}

# Helper function to atomically increment a file-based counter
increment_counter() {
    local counter_file="$1"
    # Use flock for atomic increment
    (
        flock -x 200
        local current=$(cat "$counter_file" 2>/dev/null || echo 0)
        echo $((current + 1)) > "$counter_file"
    ) 200>"${counter_file}.lock"
}

# Helper to create temp file with error checking
create_temp_file() {
    local prefix="$1"
    local suffix="${2:-.tmp}"
    local temp_file
    if ! temp_file=$(mktemp "${TEMP_DIR}/${prefix}.XXXXXX${suffix}"); then
        echo "ERROR: Failed to create temp file: ${prefix}" >&2
        return 1
    fi
    chmod 600 "$temp_file"  # Restrict to owner only
    echo "$temp_file"
}

# Helper to calculate wait seconds until rate limit reset
calculate_rate_limit_wait() {
    local reset_time_str="$1"
    local timezone="$2"

    # Validate timezone against whitelist
    if ! echo "$timezone" | grep -qE "^($ALLOWED_TIMEZONES)$"; then
        timezone="America/Los_Angeles"
    fi

    # Convert reset time to seconds since epoch
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

        local today=$(TZ="$timezone" date +%Y-%m-%d)
        if date --version >/dev/null 2>&1; then
            reset_time=$(TZ="$timezone" date -d "$today $hour:$min:00" +%s 2>/dev/null || echo "")
        else
            reset_time=$(TZ="$timezone" date -j -f "%Y-%m-%d %H:%M:%S" "$today $hour:$min:00" +%s 2>/dev/null || echo "")
        fi

        # If reset time is in the past, it might be tomorrow
        local now=$(date +%s)
        if [[ -n "$reset_time" && "$reset_time" -lt "$now" ]]; then
            reset_time=$((reset_time + 86400))
        fi
    fi

    if [[ -z "$reset_time" ]]; then
        # Default to 60 minutes if we can't parse
        echo 3600
        return
    fi

    local now=$(date +%s)
    local remaining=$((reset_time - now))
    # Minimum 60 seconds, maximum 3600 seconds
    if [[ $remaining -lt 60 ]]; then
        remaining=60
    elif [[ $remaining -gt 3600 ]]; then
        remaining=3600
    fi
    echo "$remaining"
}

# Helper to handle rate limit check
handle_rate_limit_if_found() {
    local file="$1"
    if [[ -f "$file" ]] && limit_info=$(check_rate_limit "$file"); then
        local reset_time=$(echo "$limit_info" | cut -d'|' -f1)
        local timezone=$(echo "$limit_info" | cut -d'|' -f2)
        wait_for_rate_limit_reset "$reset_time" "$timezone"
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
DRY_RUN=false
AUTO_CLEANUP=false
PARALLEL_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                echo "ERROR: --limit must be a positive integer, got '$2'"
                exit 1
            fi
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
            # Validate replan reason doesn't contain dangerous shell characters
            if [[ "$REPLAN_REASON" =~ [\;\|\&\$\`\<\>] ]]; then
                echo "ERROR: --replan-reason contains unsafe shell characters"
                exit 1
            fi
            shift 2
            ;;
        --issues)
            SPECIFIC_ISSUES="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --cleanup)
            AUTO_CLEANUP=true
            shift
            ;;
        --parallel)
            PARALLEL_MODE=true
            shift
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
            echo "  --dry-run            Preview what will be done without posting to GitHub"
            echo "  --cleanup            Delete temp directory on successful completion"
            echo "  --parallel           Process issues in parallel (requires --auto, MAX_PARALLEL=$MAX_PARALLEL)"
            echo ""
            echo "Environment Variables:"
            echo "  MAX_PARALLEL         Max concurrent jobs (default: 4)"
            echo "  CLAUDE_TIMEOUT       Timeout per issue in seconds (default: 600)"
            echo ""
            echo "Examples:"
            echo "  plan-issues.sh --limit 5                    # First 5 open issues"
            echo "  plan-issues.sh --issues 123,456,789         # Specific issues only"
            echo "  plan-issues.sh --auto --replan              # Auto mode, allow replanning"
            echo "  plan-issues.sh --issues 123 --replan-reason 'Need to add error handling'"
            echo "  plan-issues.sh --issues 123 --dry-run       # Preview without posting"
            echo "  plan-issues.sh --auto --parallel            # Parallel processing"
            echo "  MAX_PARALLEL=8 plan-issues.sh --auto --parallel  # 8 concurrent jobs"
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

# Validate chief architect file exists
CHIEF_ARCHITECT_FILE="$REPO_ROOT/.claude/agents/chief-architect.md"
if [[ ! -f "$CHIEF_ARCHITECT_FILE" ]]; then
    echo "ERROR: Chief Architect prompt not found: $CHIEF_ARCHITECT_FILE"
    exit 1
fi
CHIEF_ARCHITECT_PROMPT=$(cat "$CHIEF_ARCHITECT_FILE")

# Create secure temp directory for this run
TEMP_DIR=$(mktemp -d -t plan-issues-XXXXXX) || {
    echo "ERROR: Failed to create temp directory"
    exit 1
}
chmod 700 "$TEMP_DIR"  # Restrict to owner only

# Cleanup trap - conditionally remove temp directory
cleanup() {
    local exit_code=$?
    if [[ -n "${TEMP_DIR:-}" ]] && [[ -d "$TEMP_DIR" ]]; then
        if $AUTO_CLEANUP && [[ $exit_code -eq 0 ]]; then
            echo ""
            echo "Cleaning up temp directory: $TEMP_DIR"
            rm -rf "$TEMP_DIR"
        else
            echo ""
            echo "Temp directory preserved: $TEMP_DIR"
        fi
    fi
}
trap cleanup EXIT INT TERM

# Validate --parallel requires --auto
if $PARALLEL_MODE && ! $AUTO_MODE; then
    echo "ERROR: --parallel requires --auto mode (interactive editor not supported in parallel)"
    exit 1
fi

# Validate bash 4.3+ for parallel mode
if $PARALLEL_MODE; then
    if [[ "${BASH_VERSINFO[0]}" -lt 4 ]] || \
       [[ "${BASH_VERSINFO[0]}" -eq 4 && "${BASH_VERSINFO[1]}" -lt 3 ]]; then
        echo "ERROR: --parallel requires bash 4.3+ for 'wait -n' (found ${BASH_VERSION})"
        echo "On macOS, install with: brew install bash"
        exit 1
    fi
fi

echo "Temp directory: $TEMP_DIR"
if $AUTO_MODE; then
    echo "Mode: AUTO (non-interactive, plans auto-posted)"
else
    echo "Mode: INTERACTIVE (vim review before posting)"
fi
if $PARALLEL_MODE; then
    echo "Parallel: ENABLED (MAX_PARALLEL=$MAX_PARALLEL jobs)"
fi
if $DRY_RUN; then
    echo "Dry-run: ENABLED (no changes will be made to GitHub)"
fi
if $AUTO_CLEANUP; then
    echo "Cleanup: ENABLED (temp files deleted on success)"
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
    # Validate issue numbers are numeric
    for issue in "${issues[@]}"; do
        if ! [[ "$issue" =~ ^[0-9]+$ ]]; then
            echo "ERROR: Invalid issue number '$issue' - must be numeric"
            exit 1
        fi
    done
    echo "Issues: Using specific issues: ${issues[*]}"
else
    # Get all open issue numbers into array
    mapfile -t all_issues < <(gh issue list --state open --limit "$MAX_ISSUES_FETCH" --json number --jq '.[].number' | sort -n)

    # Apply limit if specified
    if [[ -n "$LIMIT" ]]; then
        issues=("${all_issues[@]:0:$LIMIT}")
    else
        issues=("${all_issues[@]}")
    fi
fi

# Handle empty issue list
if [[ ${#issues[@]} -eq 0 ]]; then
    echo "No issues found to process"
    exit 0
fi

total=${#issues[@]}
current=0
skipped=0
posted=0

# Initialize file-based counters for parallel mode
echo "0" > "$TEMP_DIR/posted_count"
echo "0" > "$TEMP_DIR/skipped_count"

echo "=========================================="
echo "  Issue Planning Script"
echo "  Total open issues: $total"
if $PARALLEL_MODE; then
    echo "  Parallel jobs: $MAX_PARALLEL"
fi
echo "=========================================="
echo ""

# Function to process a single issue
# Returns: 0 = posted, 1 = skipped, 2 = error
process_single_issue() {
    local issue_number="$1"
    local idx="$2"
    local total="$3"
    local output_file="$TEMP_DIR/issue-${issue_number}-output.log"
    local result_file="$TEMP_DIR/issue-${issue_number}-result.txt"

    # In parallel mode, redirect all output to file
    # Note: In parallel mode, process runs in subshell so FDs are isolated,
    # but we still save/restore for robustness and potential future direct calls
    if $PARALLEL_MODE; then
        exec 3>&1 4>&2  # Save original stdout/stderr
        exec > "$output_file" 2>&1
        # Set trap to restore FDs on any exit from function
        trap 'exec 1>&3 2>&4; exec 3>&- 4>&-' RETURN
    fi

    # Fetch all issue data in one API call
    local issue_data
    if ! issue_data=$(gh issue view "$issue_number" --json title,body,comments 2>&1); then
        echo "[$idx/$total] Issue #${issue_number}: ERROR"
        echo "----------------------------------------"
        echo "  ERROR: Failed to fetch issue: $issue_data"
        echo "skipped" > "$result_file"
        return 1
    fi

    # Extract issue data with error handling and validation
    local issue_title
    if ! issue_title=$(echo "$issue_data" | jq -r '.title // empty'); then
        echo "  ERROR: Failed to extract issue title"
        echo "skipped" > "$result_file"
        return 1
    fi
    if [[ -z "$issue_title" ]]; then
        issue_title="Untitled"
    fi

    # Validate title length
    if [[ ${#issue_title} -gt 500 ]]; then
        echo "  WARNING: Title unusually long (${#issue_title} chars), truncating to 500"
        issue_title="${issue_title:0:500}..."
    fi

    local issue_body
    if ! issue_body=$(echo "$issue_data" | jq -r '.body // ""'); then
        echo "  ERROR: Failed to extract issue body"
        echo "skipped" > "$result_file"
        return 1
    fi

    # Validate body size
    local body_size=${#issue_body}
    readonly MAX_BODY_SIZE=1048576  # 1MB limit
    if [[ $body_size -gt $MAX_BODY_SIZE ]]; then
        echo "  ERROR: Issue body too large ($body_size bytes, max $MAX_BODY_SIZE), skipping"
        echo "skipped" > "$result_file"
        return 1
    fi

    echo "[$idx/$total] Issue #${issue_number}: ${issue_title}"
    echo "----------------------------------------"

    # Check if issue already has a plan unless in replan mode
    # Performance optimization P2: Use jq filtering instead of grep on concatenated strings
    if ! $REPLAN_MODE; then
        local plan_status
        plan_status=$(echo "$issue_data" | jq -r '
            [.comments[].body // "" | select(contains("## Detailed Implementation Plan"))]
            | if length > 0 then
                if .[0] | contains("Limit reached") then "rate_limited"
                else "has_plan"
                end
              else "no_plan"
              end
        ')
        case "$plan_status" in
            has_plan)
                echo "  SKIPPED (already has plan - use --replan to override)"
                echo "skipped" > "$result_file"
                return 1
                ;;
            rate_limited)
                echo "  Existing plan was rate-limited, will replan..."
                ;;
            # no_plan: continue processing
        esac
    fi

    # Create secure temp files with error checking
    local plan_file log_file cmd_file prompt_file system_prompt_file
    if ! plan_file=$(create_temp_file "issue-${issue_number}-plan" ".md"); then
        echo "skipped" > "$result_file"
        return 1
    fi
    if ! log_file=$(create_temp_file "issue-${issue_number}-claude" ".log"); then
        echo "skipped" > "$result_file"
        return 1
    fi
    if ! cmd_file=$(create_temp_file "issue-${issue_number}-command" ".sh"); then
        echo "skipped" > "$result_file"
        return 1
    fi
    if ! prompt_file=$(create_temp_file "issue-${issue_number}-prompt" ".txt"); then
        echo "skipped" > "$result_file"
        return 1
    fi
    if ! system_prompt_file=$(create_temp_file "issue-${issue_number}-system" ".txt"); then
        echo "skipped" > "$result_file"
        return 1
    fi

    echo "  Plan file: $plan_file"
    echo "  Log file:  $log_file"

    # Write system prompt to file to avoid shell injection
    cat "$CHIEF_ARCHITECT_FILE" > "$system_prompt_file"

    # Build the prompt safely by writing to file
    cat > "$prompt_file" << 'PROMPT_TEMPLATE_EOF'
Create a detailed implementation plan for the following GitHub issue:

PROMPT_TEMPLATE_EOF

    # Append dynamic content safely
    {
        echo "Issue #${issue_number}: ${issue_title}"
        echo ""
        echo "$issue_body"
    } >> "$prompt_file"

    # Add replan instructions if applicable
    if $REPLAN_MODE; then
        {
            echo ""
            echo "NOTE: This is a REPLAN request. A previous plan exists for this issue."
            if [[ -n "$REPLAN_REASON" ]]; then
                echo "REPLAN REASON: ${REPLAN_REASON}"
                echo ""
                echo "Please review the existing plan comments on this issue and create an updated plan that addresses the replan reason."
            else
                echo ""
                echo "Please review the existing plan comments on this issue and create an improved/updated plan."
            fi
        } >> "$prompt_file"
    fi

    # Add budget and output format instructions
    cat >> "$prompt_file" << 'PROMPT_FOOTER_EOF'

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
- Steps used: Y/50
PROMPT_FOOTER_EOF

    # Set tools based on mode
    local ALLOWED_TOOLS PERMISSION_MODE
    if $AUTO_MODE; then
        ALLOWED_TOOLS="Read,Glob,Grep,WebFetch,WebSearch,Bash"
    else
        ALLOWED_TOOLS="Read,Glob,Grep,WebFetch,WebSearch"
    fi
    PERMISSION_MODE="default"

    # Save command for debugging
    cat > "$cmd_file" << CMDEOF
# Security: Using file-based input to prevent shell injection
# Prompt file: $prompt_file
# System prompt file: $system_prompt_file
claude --model opus \\
       --permission-mode $PERMISSION_MODE \\
       --allowedTools "$ALLOWED_TOOLS" \\
       --add-dir "$REPO_ROOT" \\
       --system-prompt "\$(cat $system_prompt_file)" \\
       -p "\$(cat $prompt_file)"
CMDEOF

    # Generate plan using Claude with opus and chief architect prompt
    # Retry loop for rate limiting with max retries
    local retry_count=0
    local generation_success=false

    while true; do
        retry_count=$((retry_count + 1))
        if [[ $retry_count -gt $MAX_RETRIES ]]; then
            echo "  ERROR: Max retries ($MAX_RETRIES) exceeded for issue #$issue_number"
            echo "  Logs preserved in: $log_file"
            echo "skipped" > "$result_file"
            return 1
        fi

        if [[ $retry_count -gt 1 ]]; then
            # Exponential backoff: 2^(retry-1) seconds (2s, 4s, 8s)
            local delay=$((2 ** (retry_count - 1)))
            echo "  Retry attempt $retry_count of $MAX_RETRIES (after ${delay}s backoff)..."
            sleep "$delay"
        fi

        echo "  Generating plan with Claude Opus (timeout: ${CLAUDE_TIMEOUT}s)..."
        echo "  (This may take 1-3 minutes. Check $log_file for progress)"
        local start_time end_time
        start_time=$(date +%s)

        # Run Claude - using file-based input to avoid shell injection
        echo ""
        echo "  -------- Claude Output --------"
        local PROMPT_CONTENT SYSTEM_PROMPT_CONTENT
        PROMPT_CONTENT=$(cat "$prompt_file")
        SYSTEM_PROMPT_CONTENT=$(cat "$system_prompt_file")
        # Use timeout to prevent indefinite hangs
        local claude_exit_code
        timeout "$CLAUDE_TIMEOUT" claude --model opus \
               --permission-mode "$PERMISSION_MODE" \
               --allowedTools "$ALLOWED_TOOLS" \
               --add-dir "$REPO_ROOT" \
               --system-prompt "$SYSTEM_PROMPT_CONTENT" \
               -p "$PROMPT_CONTENT" 2>&1 | tee "$plan_file"
        claude_exit_code=${PIPESTATUS[0]}
        echo "  --------------------------------"
        echo ""

        # Check for timeout (exit code 124)
        if [[ $claude_exit_code -eq 124 ]]; then
            echo "  ERROR: Claude generation timed out after ${CLAUDE_TIMEOUT}s"
            echo "  Retrying..."
            continue
        fi

        end_time=$(date +%s)
        echo "  Generation completed in $((end_time - start_time))s"
        echo "  Plan size: $(wc -c < "$plan_file") bytes"

        # Check for Claude CLI errors (JSON error response)
        if grep -q '"type":"result","subtype":"error' "$plan_file" 2>/dev/null; then
            echo "  ERROR: Claude CLI returned an error response"
            local error_msgs
            error_msgs=$(grep -o '"errors":\[[^]]*\]' "$plan_file" | head -1)
            if [[ -n "$error_msgs" ]]; then
                echo "  Error details: $error_msgs"
            fi
            echo "  Retrying..."
            continue
        fi

        # Check for rate limit in output using helper
        # Note: In parallel mode, rate limit handling calculates actual wait time
        if $PARALLEL_MODE; then
            local limit_info
            if limit_info=$(check_rate_limit "$plan_file" 2>/dev/null) || \
               limit_info=$(check_rate_limit "$log_file" 2>/dev/null); then
                local reset_time=$(echo "$limit_info" | cut -d'|' -f1)
                local timezone=$(echo "$limit_info" | cut -d'|' -f2)
                local wait_seconds=$(calculate_rate_limit_wait "$reset_time" "$timezone")
                echo "  Rate limit detected, waiting ${wait_seconds}s until reset..."
                sleep "$wait_seconds"
                continue
            fi
        else
            if handle_rate_limit_if_found "$plan_file" || handle_rate_limit_if_found "$log_file"; then
                echo "  Retrying plan generation..."
                continue
            fi
        fi

        # No errors or rate limit, mark success and break
        generation_success=true
        break
    done

    # Return if generation failed
    if ! $generation_success; then
        echo "skipped" > "$result_file"
        return 1
    fi

    # Interactive mode: open editor for review only in sequential mode
    if ! $AUTO_MODE; then
        local EDIT_CMD EDIT_CMD_PATH
        EDIT_CMD=$(basename "${EDITOR:-vim}")

        # Validate against whitelist
        if [[ ! "$EDIT_CMD" =~ ^($APPROVED_EDITORS)$ ]]; then
            echo "  WARNING: EDITOR '$EDIT_CMD' not in approved list, falling back to vim"
            EDIT_CMD="vim"
        fi

        # Get absolute path to prevent PATH manipulation attacks
        if ! EDIT_CMD_PATH=$(command -v "$EDIT_CMD" 2>/dev/null); then
            echo "  WARNING: Editor '$EDIT_CMD' not found, trying vim..."
            EDIT_CMD="vim"
            if ! EDIT_CMD_PATH=$(command -v "vim" 2>/dev/null); then
                echo "  ERROR: Neither '$EDIT_CMD' nor 'vim' found in PATH"
                echo "skipped" > "$result_file"
                return 1
            fi
        fi

        # Verify it's an executable file
        if [[ ! -x "$EDIT_CMD_PATH" ]]; then
            echo "  ERROR: Editor path '$EDIT_CMD_PATH' is not executable"
            echo "skipped" > "$result_file"
            return 1
        fi

        echo "  Opening $EDIT_CMD for review... (delete all content to skip)"
        "$EDIT_CMD_PATH" "$plan_file"

        # Check if file is empty which means the user wants to skip filing the plan
        if [[ ! -s "$plan_file" ]]; then
            echo "  SKIPPED (empty file)"
            echo "skipped" > "$result_file"
            return 1
        fi
    fi

    # Check if plan was generated
    if [[ ! -s "$plan_file" ]]; then
        echo "  ERROR: No plan generated (empty file)"
        echo "skipped" > "$result_file"
        return 1
    fi

    # Build plan header
    local PLAN_HEADER
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

    # Post plan to GitHub issue
    if $DRY_RUN; then
        echo "  [DRY RUN] Would post plan to issue #$issue_number"
        echo "  Plan preview (first 20 lines):"
        head -20 "$plan_file" | sed 's/^/    /'
        echo "posted" > "$result_file"
    else
        echo "  Posting plan to GitHub..."
        # Use --body-file to avoid command injection
        # Check result to avoid silent failures
        if ! {
            echo "$PLAN_HEADER"
            cat "$plan_file"
        } | gh issue comment "$issue_number" --body-file -; then
            echo "  ERROR: Failed to post plan to GitHub"
            echo "skipped" > "$result_file"
            return 1
        fi
        echo "posted" > "$result_file"
        echo "  Plan posted successfully!"
    fi
    echo ""
    return 0
}

# Export variables needed by the function in subshells
export TEMP_DIR REPO_ROOT CHIEF_ARCHITECT_FILE AUTO_MODE REPLAN_MODE REPLAN_REASON
export DRY_RUN PARALLEL_MODE MAX_RETRIES CLAUDE_TIMEOUT APPROVED_EDITORS
export LOG_PREFIX_INFO LOG_PREFIX_WARN LOG_PREFIX_ERROR LOG_PREFIX_DEBUG
export ALLOWED_TIMEZONES

# Export helper functions needed by subshells in parallel mode
export -f log_info
export -f log_warn
export -f log_error
export -f log_debug
export -f check_rate_limit
export -f handle_rate_limit_if_found
export -f wait_for_rate_limit_reset
export -f calculate_rate_limit_wait
export -f increment_counter
export -f create_temp_file
export -f process_single_issue

# Process issues - parallel or sequential based on mode
if $PARALLEL_MODE; then
    echo "Starting parallel processing with $MAX_PARALLEL concurrent jobs..."
    echo ""

    # Create semaphore directory for slot-based concurrency control
    SEMAPHORE_DIR="$TEMP_DIR/semaphores"
    mkdir -p "$SEMAPHORE_DIR"

    # Create slot files for flock-based semaphore
    for ((slot=0; slot<MAX_PARALLEL; slot++)); do
        touch "$SEMAPHORE_DIR/slot-$slot"
    done

    # Function to acquire a semaphore slot which blocks until available
    acquire_slot() {
        local slot_acquired=false
        while ! $slot_acquired; do
            for ((slot=0; slot<MAX_PARALLEL; slot++)); do
                # Try to acquire this slot with non-blocking flock
                if (
                    flock -n 200 || exit 1
                    # Hold lock for duration of subshell
                    "$@"
                ) 200>"$SEMAPHORE_DIR/slot-$slot" 2>/dev/null; then
                    slot_acquired=true
                    break
                fi
            done
            # If no slot available, wait briefly before retrying
            if ! $slot_acquired; then
                sleep 0.1
            fi
        done
    }

    # Track background jobs
    declare -a job_pids=()
    declare -a job_issues=()
    idx=0

    for issue_number in "${issues[@]}"; do
        idx=$((idx + 1))

        # Wait until a slot is available using flock
        # Find first available slot
        while true; do
            slot_found=false
            for ((slot=0; slot<MAX_PARALLEL; slot++)); do
                slot_file="$SEMAPHORE_DIR/slot-$slot"
                # Try non-blocking lock on this slot
                if ! flock -n "$slot_file" true 2>/dev/null; then
                    # Slot is taken, try next
                    continue
                fi
                # Slot is free, we can use it
                slot_found=true
                break
            done
            if $slot_found; then
                break
            fi
            # All slots taken, wait for any job to finish
            wait -n 2>/dev/null || true
            # Clean up finished jobs from tracking
            new_pids=() new_issues=()
            for i in "${!job_pids[@]}"; do
                if kill -0 "${job_pids[$i]}" 2>/dev/null; then
                    new_pids+=("${job_pids[$i]}")
                    new_issues+=("${job_issues[$i]}")
                fi
            done
            job_pids=("${new_pids[@]:-}")
            job_issues=("${new_issues[@]:-}")
        done

        # Launch job with slot lock held for duration
        (
            flock -x 200
            process_single_issue "$issue_number" "$idx" "$total"
        ) 200>"$SEMAPHORE_DIR/slot-$slot" &
        job_pids+=($!)
        job_issues+=("$issue_number")

        echo "  Started job for issue #$issue_number (PID: ${job_pids[-1]}, slot: $slot, $idx/$total)"
    done

    # Wait for all remaining jobs to complete
    echo ""
    echo "Waiting for remaining jobs to complete..."
    wait

    # Aggregate results from result files
    echo ""
    echo "Aggregating results..."
    failed_issues=()
    for issue_number in "${issues[@]}"; do
        result_file="$TEMP_DIR/issue-${issue_number}-result.txt"
        output_file="$TEMP_DIR/issue-${issue_number}-output.log"

        # Print the output from each job
        if [[ -f "$output_file" ]]; then
            cat "$output_file"
        fi

        # Count results
        if [[ -f "$result_file" ]]; then
            result=$(cat "$result_file")
            if [[ "$result" == "posted" ]]; then
                posted=$((posted + 1))
            else
                skipped=$((skipped + 1))
            fi
        else
            # Job crashed or was killed without writing result
            skipped=$((skipped + 1))
            failed_issues+=("$issue_number")
        fi
    done

    # Report any crashed jobs
    if [[ ${#failed_issues[@]} -gt 0 ]]; then
        echo ""
        echo "WARNING: ${#failed_issues[@]} job(s) failed without writing results:"
        printf "  Issue #%s\n" "${failed_issues[@]}"
        echo "Check log files in $TEMP_DIR for details"
    fi
else
    # Sequential processing
    for issue_number in "${issues[@]}"; do
        current=$((current + 1))

        if process_single_issue "$issue_number" "$current" "$total"; then
            posted=$((posted + 1))
        else
            skipped=$((skipped + 1))
        fi
    done
fi

echo "=========================================="
echo "  Summary"
echo "  Posted: $posted"
echo "  Skipped: $skipped"
echo "  Total: $total"
echo ""
if ! $AUTO_CLEANUP; then
    echo "  Temp directory: $TEMP_DIR"
    echo "  - Plans: issue-*-plan.*.md"
    echo "  - Logs:  issue-*-claude.*.log"
    echo "  - Cmds:  issue-*-command.*.sh"
fi
echo "=========================================="
