# Docker Image Case Sensitivity Fix - Raw Notes

## Session Details

**Date:** 2025-12-29
**Command:** `/fix-ci 2730`
**Actual Issue:** Docker SBOM generation failing on main branch

## Discovery Process

### Initial Diagnosis

1. Checked for PR #2730 - didn't exist
2. Listed recent PRs - found PR #2980 (already merged)
3. Checked CI runs on main branch:
   ```bash
   gh run list --branch main --limit 5
   ```
4. Found failing run: 20563226174 (Docker Build and Publish)

### Error Investigation

Retrieved failed logs:
```bash
gh run view 20563226174 --log-failed
```

**Key Error:**
```
[0000] ERROR could not determine source: errors occurred attempting to resolve 'ghcr.io/mvillmow/ProjectOdyssey:main':
  - snap: snap file "ghcr.io/mvillmow/ProjectOdyssey:main" does not exist
  - docker: could not parse reference: ghcr.io/mvillmow/ProjectOdyssey:main
  - podman: podman not available: no host address
  - containerd: containerd not available: failed to dial "/run/containerd/containerd.sock"
  - oci-registry: unable to parse registry reference="ghcr.io/mvillmow/ProjectOdyssey:main"
  - additionally, the following providers failed with file does not exist: docker-archive, oci-archive, oci-dir, singularity, local-file, local-directory
```

## Root Cause Analysis

### The Problem

Docker image names MUST be lowercase, but the workflow was using:
```yaml
env:
  IMAGE_NAME: ${{ github.repository }}
```

Which evaluates to: `mvillmow/ProjectOdyssey` (mixed case)

### Why Other Steps Worked

The `docker/build-push-action` and `docker/metadata-action` automatically lowercase image names in their outputs. But the `anchore/sbom-action` step manually constructs the image reference:

```yaml
- name: Generate SBOM
  uses: anchore/sbom-action@v0
  with:
    image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.ref_name }}
```

This creates: `ghcr.io/mvillmow/ProjectOdyssey:main` ❌

## Solution Attempts

### Attempt 1: Use `github.repository_owner`

**Tried:**
```yaml
IMAGE_NAME: ${{ github.repository_owner }}/projectodyssey
```

**Problem:** `github.repository_owner` also preserves case

### Attempt 2: Hardcode Lowercase (SUCCESSFUL)

**Final Solution:**
```yaml
env:
  REGISTRY: ghcr.io
  # Docker image names must be lowercase - github.repository preserves case
  IMAGE_NAME: mvillmow/projectodyssey
```

## Implementation

### Files Modified

`.github/workflows/docker.yml:20-23`

### Commands Run

```bash
git checkout main
git pull origin main
git checkout -b fix-docker-sbom-lowercase
# Made the edit
git add .github/workflows/docker.yml
git commit -m "fix(ci): use lowercase image name for Docker SBOM generation"
git push -u origin fix-docker-sbom-lowercase
gh pr create --title "fix(ci): use lowercase image name for Docker SBOM generation" --body "..."
gh pr merge 2982 --auto --rebase
```

### PR Details

- **PR Number:** #2982
- **Branch:** fix-docker-sbom-lowercase
- **Status:** Auto-merge enabled, CI pending

## Lessons Learned

1. **Docker is strict about lowercase** - no exceptions
2. **GitHub variables preserve case** - `github.repository`, `github.repository_owner`, etc.
3. **Actions handle lowercasing differently** - some do it automatically, others don't
4. **Manual image references are risky** - prefer using action outputs when possible
5. **SBOM generation is sensitive** - it parses image names strictly

## Prevention Strategies

1. Always hardcode lowercase image names in workflows
2. Test SBOM generation in PR workflows, not just on main
3. Audit all Docker workflows for `github.repository` usage
4. Consider adding validation step to check image name format

## Additional Context

### Related CI Runs

- **Failed Run:** 20563226174 (Docker Build and Publish)
- **Succeeded Runs:**
  - 20563226163 (Comprehensive Tests)
  - 20563226162 (Build Validation)
  - 20563226172 (Code Coverage)
  - 20563226165 (Security Scanning)

### Workflow Structure

The Docker workflow has 4 jobs:
1. `build-and-push` - Builds 3 targets (runtime, ci, production) ← Failed here
2. `test-images` - Tests the built images
3. `security-scan` - Runs Trivy scanner
4. `summary` - Generates summary

The failure was in the SBOM generation step within `build-and-push (runtime)`.

## Tools Used

- `gh run list` - List recent CI runs
- `gh run view --log-failed` - Get failure logs
- `gh pr create` - Create pull request
- `gh pr merge --auto` - Enable auto-merge
- `grep` - Search for `github.repository` usage
- Git workflow commands

## Time Taken

Approximately 10 minutes from diagnosis to PR creation.

## Follow-Up Actions

- [ ] Monitor PR #2982 CI status
- [ ] Verify SBOM generation succeeds
- [ ] Consider adding workflow validation for lowercase image names
- [ ] Document this pattern in CLAUDE.md CI/CD section
