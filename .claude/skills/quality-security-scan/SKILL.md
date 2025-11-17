---
name: quality-security-scan
description: Scan code for security vulnerabilities and unsafe patterns. Use before committing sensitive code or in security reviews.
---

# Security Scan Skill

Scan code for security vulnerabilities.

## When to Use

- Before committing code with secrets
- Security review process
- Handling sensitive data
- Pre-release security audit

## Security Checks

### 1. Secrets Detection

```bash
# Check for committed secrets
./scripts/scan_for_secrets.sh

# Detects:
# - API keys
# - Passwords
# - Private keys
# - Tokens
```

### 2. Dependency Vulnerabilities

```bash
# Check Python dependencies
pip-audit

# Check for known vulnerabilities
safety check
```

### 3. Code Patterns

```bash
# Check for unsafe patterns
./scripts/check_unsafe_patterns.sh

# Looks for:
# - Hardcoded credentials
# - SQL injection vectors
# - Unsafe file operations
# - Unvalidated input
```

## Prevention

### .gitignore

Ensure sensitive files ignored:

```text
.env
*.key
*.pem
credentials.json
secrets/
```

### Pre-commit Hook

```yaml
- id: detect-private-key
  name: Detect Private Key
- id: detect-aws-credentials
  name: Detect AWS Credentials
```

## Common Vulnerabilities

### 1. Hardcoded Secrets

```python
# ❌ Wrong
API_KEY = "sk_live_1234567890"

# ✅ Correct
import os
API_KEY = os.getenv("API_KEY")
```

### 2. Unsafe File Operations

```mojo
# ❌ Potential path traversal
fn load_file(path: String):
    open(path)

# ✅ Validate path
fn load_file(path: String):
    if is_safe_path(path):
        open(path)
```

See security best practices documentation.
