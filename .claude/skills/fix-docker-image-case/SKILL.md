# Skill: Fix Docker Image Case Sensitivity in CI/CD

| Property | Value |
|----------|-------|
| **Date** | 2025-12-29 |
| **Category** | ci-cd |
| **Objective** | Fix Docker SBOM generation failures caused by mixed-case image names |
| **Outcome** | ✅ Successfully fixed by hardcoding lowercase image name |
| **Context** | GitHub Actions workflow using `anchore/sbom-action` |

## When to Use

Invoke this skill when:

1. **SBOM generation fails** with errors like:
   - `could not parse reference: ghcr.io/Owner/RepoName:tag`
   - `unable to parse registry reference`
   - Docker image reference parsing errors in CI

2. **GitHub Actions Docker workflows fail** with case-related errors

3. **Using `github.repository` variable** in Docker image references

4. **After repository renames** that change capitalization

## Problem Overview

### Root Cause

Docker image names **must be lowercase**, but GitHub's `github.repository` variable preserves the original repository name case (e.g., `mvillmow/ProjectOdyssey`). This causes failures in tools that manually construct Docker image references (like `anchore/sbom-action`).

### Why `docker/metadata-action` Works

The `docker/metadata-action` automatically lowercases image names in its outputs, so build/push steps work fine. However, when **manually constructing image references** using `${{ env.IMAGE_NAME }}`, the original case is preserved, causing failures.

## Verified Workflow

### Step 1: Identify the Failure

Check CI logs for image parsing errors:

```bash
gh run list --branch main --limit 5
gh run view <run-id> --log-failed
```

Look for errors like:
```
could not parse reference: ghcr.io/mvillmow/ProjectOdyssey:main
```

### Step 2: Locate Mixed-Case References

Search the workflow file for uses of `github.repository`:

```bash
grep -n "github.repository" .github/workflows/docker.yml
```

### Step 3: Fix Environment Variable

Replace dynamic repository reference with hardcoded lowercase:

```yaml
# ❌ BEFORE (preserves case)
env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

# ✅ AFTER (hardcoded lowercase)
env:
  REGISTRY: ghcr.io
  # Docker image names must be lowercase - github.repository preserves case
  IMAGE_NAME: mvillmow/projectodyssey
```

### Step 4: Verify All Image References

Check all places where `env.IMAGE_NAME` is used:
- SBOM generation steps
- Image scanning steps
- Manual docker pull/push commands
- Summary generation

### Step 5: Test the Fix

Create PR and verify CI passes:

```bash
git checkout -b fix-docker-image-case
git add .github/workflows/docker.yml
git commit -m "fix(ci): use lowercase image name for Docker operations"
git push -u origin fix-docker-image-case
gh pr create --title "fix(ci): use lowercase image name" --body "Fixes Docker SBOM generation"
gh pr merge --auto --rebase
```

## Failed Attempts

### ❌ Attempt 1: Use `github.repository_owner`

**Tried:**
```yaml
IMAGE_NAME: ${{ github.repository_owner }}/projectodyssey
```

**Why it failed:**
- `github.repository_owner` also preserves case (e.g., `mVillmow` vs `mvillmow`)
- GitHub context variables don't automatically lowercase

### ❌ Attempt 2: Rely on `docker/metadata-action` Everywhere

**Why it didn't work:**
- The `docker/metadata-action` only lowercases its **own outputs** (`steps.meta.outputs.tags`)
- Other actions (like `anchore/sbom-action`) that use `env.IMAGE_NAME` don't get the lowercased version
- Manual image references in workflow scripts also fail

## Results & Parameters

### Successful Fix

**File:** `.github/workflows/docker.yml`

**Change:**
```yaml
env:
  REGISTRY: ghcr.io
  IMAGE_NAME: mvillmow/projectodyssey  # Hardcoded lowercase
```

**Affected Steps:**
- `Generate SBOM` (anchore/sbom-action)
- `Test runtime image` (docker run commands)
- `Test production image` (docker run commands)
- `Run Trivy vulnerability scanner`
- Summary generation

### Error Before Fix

```
[0000] ERROR could not determine source: errors occurred attempting to resolve 'ghcr.io/mvillmow/ProjectOdyssey:main':
  - docker: could not parse reference: ghcr.io/mvillmow/ProjectOdyssey:main
  - oci-registry: unable to parse registry reference="ghcr.io/mvillmow/ProjectOdyssey:main"
```

### Success After Fix

- SBOM generation completes successfully
- Image scanning works
- All Docker operations use consistent lowercase names

## Prevention

### For New Workflows

1. **Always hardcode lowercase** image names in `env.IMAGE_NAME`
2. **Never use** `github.repository` directly in Docker contexts
3. **Test SBOM generation** in PR workflow (not just main)

### For Existing Workflows

1. **Audit all Docker workflows** for `github.repository` usage
2. **Search for image references**: `grep -r "IMAGE_NAME" .github/workflows/`
3. **Add validation**: Test lowercase conversion in workflow

## Related Skills

- `fix-ci-failures` - General CI failure diagnosis
- `analyze-ci-failure-logs` - Log parsing and root cause analysis
- `validate-workflow` - GitHub Actions workflow validation

## References

- **PR #2982**: Initial fix for Docker SBOM lowercase issue
- **Run 20563226174**: Failed Docker build with case error
- **Error Log**: Shows all providers failing to parse mixed-case reference

## Notes

- This issue is **specific to manual image references** - build/push steps work fine
- The fix is **intentionally hardcoded** to avoid GitHub variable case issues
- Consider using **organization/repository naming conventions** that are already lowercase
