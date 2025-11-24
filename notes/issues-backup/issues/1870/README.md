# Issue #1870: Implement Smart Rate Limiting for GitHub API Calls

## Overview

Replace hardcoded `time.sleep(2)` with intelligent rate limiting based on actual GitHub API rate limit status.

## Current Behavior

`scripts/create_issues.py` line 649 uses hardcoded 2-second sleep after every API call, which is inefficient and slows down issue creation unnecessarily.

## Proposed Solution

Implement `smart_rate_limit_sleep()` function that:

- Checks actual GitHub API rate limit via `gh api rate_limit`
- Sleeps only when necessary based on remaining calls
- Uses exponential backoff when rate limit is low
- No sleep when rate limit is healthy (>100 remaining)

## Implementation Approach

```python
def check_github_rate_limit() -> Tuple[int, float]:
    """Check GitHub API rate limit status."""
    result = subprocess.run(
        ["gh", "api", "rate_limit"],
        capture_output=True,
        text=True
    )
    data = json.loads(result.stdout)
    remaining = data["resources"]["core"]["remaining"]
    reset_time = data["resources"]["core"]["reset"]
    return remaining, reset_time

def smart_rate_limit_sleep():
    """Sleep only if necessary based on rate limit."""
    remaining, reset_time = check_github_rate_limit()

    if remaining < 10:
        # Critical - wait until reset
        wait_time = max(0, reset_time - time.time())
        if wait_time > 0:
            logging.warning(f"Rate limit critical, waiting {wait_time:.0f}s")
            time.sleep(min(wait_time, 60))
    elif remaining < 100:
        # Low - exponential backoff
        backoff = (100 - remaining) / 20
        time.sleep(backoff)
    # Otherwise no sleep - rate limit is healthy
```

## Benefits

- Faster issue creation when rate limit is healthy
- Intelligent backoff when rate limit is low
- Prevents rate limit exhaustion
- Better user experience

## Implementation

### Changes Made

1. **Added check_github_rate_limit() function** (lines 67-94)
   - Calls `gh api rate_limit` to get current status
   - Returns (remaining calls, reset timestamp)
   - Handles errors gracefully with conservative fallback values

1. **Added smart_rate_limit_sleep() function** (lines 97-126)
   - No sleep if remaining > 100 (healthy)
   - Exponential backoff if 10 < remaining ≤ 100 (low)
   - Wait until reset if remaining ≤ 10 (critical, max 60s)
   - Logs rate limit status for visibility

1. **Replaced hardcoded sleep** (line 711)
   - Changed from `time.sleep(2)` to `smart_rate_limit_sleep()`
   - Now adapts to actual API conditions

### Testing Results

```bash
# Test rate limit checking
✓ Rate limit check successful
  Remaining calls: 5000
  Reset time: 1763753124
  Rate limit is: HEALTHY

# Test smart sleep function
✓ Function executed successfully
  Time elapsed: 0.462s (API call only, no sleep with healthy rate limit)
```

### Performance Impact

**Before**: Fixed 2-second sleep after every issue

- 100 issues = 200 seconds (3m 20s) of pure waiting

**After**: Adaptive sleep based on rate limit

- Healthy rate limit (>100 remaining): ~0.5s API check only
- 100 issues = 50 seconds (0m 50s) of API checks
- **70% faster** when rate limit is healthy!

Low rate limit still prevents exhaustion with intelligent backoff.

## Status

**COMPLETED** ✅ - Implemented and tested

## Related Issues

Part of Wave 2 tooling improvements from continuous improvement session.
