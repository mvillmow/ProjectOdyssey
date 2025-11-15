# CI/CD Pipeline Details

Complete reference for ML Odyssey's CI/CD infrastructure including workflow configurations, troubleshooting,
and advanced patterns.

> **Quick Reference**: For a concise overview, see [CI/CD Pipeline](../dev/ci-cd.md).

## Table of Contents

- [Complete Workflow Configurations](#complete-workflow-configurations)
- [Advanced GitHub Actions Patterns](#advanced-github-actions-patterns)
- [Performance Optimization](#performance-optimization)
- [Security and Secrets Management](#security-and-secrets-management)
- [Debugging CI Failures](#debugging-ci-failures)
- [Custom Workflows](#custom-workflows)
- [Matrix Testing](#matrix-testing)
- [Deployment Strategies](#deployment-strategies)

## Complete Workflow Configurations

### Test Workflow (Complete)

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [main]
    paths:
      - 'shared/**'
      - 'papers/**'
      - 'tests/**'
      - 'pixi.toml'
      - 'pyproject.toml'
  pull_request:
    branches: [main]
  workflow_dispatch:  # Allow manual trigger

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

env:
  PIXI_VERSION: "0.11.0"
  PYTHON_VERSION: "3.11"

jobs:
  test:
    name: Test on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    timeout-minutes: 30

    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        include:
          - os: ubuntu-latest
            cache-path: ~/.pixi
          - os: macos-latest
            cache-path: ~/Library/Caches/pixi

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for coverage

      - name: Cache Pixi environment
        uses: actions/cache@v3
        with:
          path: ${{ matrix.cache-path }}
          key: ${{ runner.os }}-pixi-${{ hashFiles('**/pixi.lock') }}
          restore-keys: |
            ${{ runner.os }}-pixi-

      - name: Install Pixi
        run: |
          curl -fsSL https://pixi.sh/install.sh | bash
          echo "$HOME/.pixi/bin" >> $GITHUB_PATH

      - name: Verify Pixi installation
        run: pixi --version

      - name: Install dependencies
        run: pixi install

      - name: Run tests
        run: pixi run pytest tests/ --cov=shared --cov=papers --cov-report=xml --cov-report=html

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-${{ matrix.os }}
          fail_ci_if_error: true

      - name: Upload coverage HTML
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: coverage-html-${{ matrix.os }}
          path: htmlcov/
          retention-days: 7

      - name: Check coverage threshold
        run: |
          COVERAGE=$(python -c "import xml.etree.ElementTree as ET; tree = ET.parse('coverage.xml'); print(tree.getroot().attrib['line-rate'])")
          echo "Coverage: $COVERAGE"
          if (( $(echo "$COVERAGE < 0.90" | bc -l) )); then
            echo "Coverage below 90% threshold"
            exit 1
          fi
```

### Pre-commit Workflow (Complete)

```yaml
# .github/workflows/pre-commit.yml
name: Pre-commit

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Cache pre-commit
        uses: actions/cache@v3
        with:
          path: ~/.cache/pre-commit
          key: ${{ runner.os }}-pre-commit-${{ hashFiles('.pre-commit-config.yaml') }}
          restore-keys: |
            ${{ runner.os }}-pre-commit-

      - name: Install pre-commit
        run: pip install pre-commit

      - name: Run pre-commit
        run: pre-commit run --all-files --show-diff-on-failure

      - name: Annotate code with errors
        if: failure()
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const output = fs.readFileSync('pre-commit-output.txt', 'utf8');
            github.rest.checks.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              name: 'pre-commit-failures',
              head_sha: context.sha,
              conclusion: 'failure',
              output: {
                title: 'Pre-commit Hook Failures',
                summary: output
              }
            });
```

### Benchmark Workflow (Complete)

```yaml
# .github/workflows/benchmark.yml
name: Benchmarks

on:
  pull_request:
    branches: [main]
    paths:
      - 'shared/**/*.mojo'
      - 'benchmarks/**'
  workflow_dispatch:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    steps:
      - name: Checkout PR
        uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}

      - name: Install Pixi
        run: |
          curl -fsSL https://pixi.sh/install.sh | bash
          echo "$HOME/.pixi/bin" >> $GITHUB_PATH

      - name: Install dependencies
        run: pixi install

      - name: Run benchmarks
        run: |
          pixi run mojo run benchmarks/run_all.mojo --output=current.json

      - name: Checkout main
        run: |
          git fetch origin main
          git checkout origin/main

      - name: Run baseline benchmarks
        run: |
          pixi install
          pixi run mojo run benchmarks/run_all.mojo --output=baseline.json

      - name: Compare results
        id: compare
        run: |
          python scripts/compare_benchmarks.py \
            --current current.json \
            --baseline baseline.json \
            --output comparison.md \
            --threshold 0.1  # Fail if >10% regression

      - name: Comment PR with results
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const comparison = fs.readFileSync('comparison.md', 'utf8');

            // Find existing comment
            const { data: comments } = await github.rest.issues.listComments({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number
            });

            const botComment = comments.find(comment =>
              comment.user.type === 'Bot' && comment.body.includes('## Benchmark Comparison')
            );

            // Update or create comment
            if (botComment) {
              await github.rest.issues.updateComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                comment_id: botComment.id,
                body: comparison
              });
            } else {
              await github.rest.issues.createComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: context.issue.number,
                body: comparison
              });
            }

      - name: Check for regressions
        run: |
          if [ -f "regression.flag" ]; then
            echo "Performance regression detected!"
            exit 1
          fi
```

## Advanced GitHub Actions Patterns

### Conditional Job Execution

```yaml
jobs:
  check-changes:
    runs-on: ubuntu-latest
    outputs:
      shared: ${{ steps.filter.outputs.shared }}
      papers: ${{ steps.filter.outputs.papers }}
      docs: ${{ steps.filter.outputs.docs }}
    steps:
      - uses: actions/checkout@v4
      - uses: dorny/paths-filter@v2
        id: filter
        with:
          filters: |
            shared:
              - 'shared/**'
            papers:
              - 'papers/**'
            docs:
              - 'docs/**'

  test-shared:
    needs: check-changes
    if: needs.check-changes.outputs.shared == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pixi run pytest tests/shared/

  test-papers:
    needs: check-changes
    if: needs.check-changes.outputs.papers == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pixi run pytest tests/papers/
```

### Reusable Workflows

```yaml
# .github/workflows/reusable-test.yml
name: Reusable Test Workflow

on:
  workflow_call:
    inputs:
      test-path:
        required: true
        type: string
      python-version:
        required: false
        type: string
        default: '3.11'
    outputs:
      coverage:
        description: "Test coverage percentage"
        value: ${{ jobs.test.outputs.coverage }}

jobs:
  test:
    runs-on: ubuntu-latest
    outputs:
      coverage: ${{ steps.coverage.outputs.value }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ inputs.python-version }}
      - run: pixi run pytest ${{ inputs.test-path }} --cov --cov-report=json
      - id: coverage
        run: echo "value=$(jq '.totals.percent_covered' coverage.json)" >> $GITHUB_OUTPUT

# Use in another workflow:
# jobs:
#   test-shared:
#     uses: ./.github/workflows/reusable-test.yml
#     with:
#       test-path: tests/shared/
```

## Performance Optimization

### Parallel Job Execution

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        test-group: [unit, integration, e2e]
        shard: [1, 2, 3, 4]
    steps:
      - uses: actions/checkout@v4
      - run: |
          pixi run pytest tests/${{ matrix.test-group }}/ \
            --shard-id=${{ matrix.shard }} \
            --num-shards=4
```

### Caching Strategies

```yaml
- name: Cache Mojo build artifacts
  uses: actions/cache@v3
  with:
    path: |
      ~/.modular
      build/
    key: ${{ runner.os }}-mojo-${{ hashFiles('shared/**/*.mojo') }}
    restore-keys: |
      ${{ runner.os }}-mojo-

- name: Cache pip packages
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

- name: Cache test results
  uses: actions/cache@v3
  with:
    path: .pytest_cache
    key: ${{ runner.os }}-pytest-${{ hashFiles('tests/**') }}
```

## Security and Secrets Management

### Secure Secret Usage

```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production  # Requires approval
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Deploy to S3
        run: |
          aws s3 sync ./site s3://ml-odyssey-docs/
        env:
          AWS_DEFAULT_REGION: us-east-1
```

### Dependency Scanning

```yaml
# .github/workflows/security.yml
name: Security Scan

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  dependency-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/dependency-review-action@v3
        with:
          fail-on-severity: moderate

  codeql:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    steps:
      - uses: actions/checkout@v4
      - uses: github/codeql-action/init@v2
        with:
          languages: python
      - uses: github/codeql-action/analyze@v2
```

## Debugging CI Failures

### Enable Debug Logging

```yaml
- name: Enable debug logging
  run: echo "ACTIONS_STEP_DEBUG=true" >> $GITHUB_ENV

- name: Debug information
  run: |
    echo "Runner OS: $RUNNER_OS"
    echo "Runner Arch: $RUNNER_ARCH"
    echo "Working Directory: $(pwd)"
    echo "Environment Variables:"
    env | sort
    echo "Disk Space:"
    df -h
    echo "Memory:"
    free -h
```

### Interactive Debugging with tmate

```yaml
- name: Setup tmate session
  uses: mxschmitt/action-tmate@v3
  if: ${{ failure() }}
  with:
    limit-access-to-actor: true
```

## Custom Workflows

### Scheduled Cleanup

```yaml
# .github/workflows/cleanup.yml
name: Cleanup Old Artifacts

on:
  schedule:
    - cron: '0 0 * * 0'  # Every Sunday
  workflow_dispatch:

jobs:
  cleanup:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/github-script@v6
        with:
          script: |
            const artifacts = await github.rest.actions.listArtifactsForRepo({
              owner: context.repo.owner,
              repo: context.repo.repo,
              per_page: 100
            });

            const oneWeekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);

            for (const artifact of artifacts.data.artifacts) {
              const createdAt = new Date(artifact.created_at);
              if (createdAt < oneWeekAgo) {
                await github.rest.actions.deleteArtifact({
                  owner: context.repo.owner,
                  repo: context.repo.repo,
                  artifact_id: artifact.id
                });
                console.log(`Deleted artifact: ${artifact.name}`);
              }
            }
```

## Matrix Testing

### Multi-Platform Testing

```yaml
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.10', '3.11', '3.12']
        mojo-version: ['24.5.0', 'nightly']
        exclude:
          - os: windows-latest
            mojo-version: 'nightly'
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - name: Set up environment
        run: |
          curl -fsSL https://pixi.sh/install.sh | bash
          pixi install
      - name: Run tests
        run: pixi run pytest tests/
```

## Deployment Strategies

### Blue-Green Deployment

```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to green environment
        run: ./scripts/deploy.sh green

      - name: Run smoke tests
        run: ./scripts/smoke-test.sh green

      - name: Switch traffic
        run: ./scripts/switch-traffic.sh green

      - name: Monitor for errors
        run: ./scripts/monitor.sh --duration=300

      - name: Rollback on failure
        if: failure()
        run: ./scripts/switch-traffic.sh blue
```

This appendix provides complete CI/CD details. For quick reference,
see [CI/CD Pipeline](../dev/ci-cd.md).
