# Critical Issue Action Items

## Priority 1: Memory & Safety Issues

### Training Module

- [ ] **TrainingState lifetime annotations**
  - Issue: Borrowed references without clear lifetimes
  - Fix: Add explicit lifetime parameters or use owned types
  - File: shared/training/base.mojo
  - Severity: CRITICAL
- [ ] **ModelCheckpoint error handling**
  - Issue: No error handling in file I/O
  - Fix: Add try/except with specific error types
  - File: shared/training/callbacks.mojo
  - Severity: CRITICAL

### Utils I/O Module

- [ ] **Complete atomic write implementation**
  - Issue: Falls back to non-atomic writes
  - Fix: Implement proper temp file + rename pattern
  - File: shared/utils/io.mojo:safe_write_file()
  - Severity: CRITICAL
- [ ] **Path traversal protection**
  - Issue: No validation in join_path
  - Fix: Add path validation and sanitization
  - File: shared/utils/io.mojo:join_path()
  - Severity: CRITICAL

### Utils Config Module

- [ ] **ConfigValue type safety**
  - Issue: Union type implementation unsafe
  - Fix: Add runtime type checking with clear errors
  - File: shared/utils/config.mojo
  - Severity: CRITICAL
- [ ] **Config file validation**
  - Issue: No input validation
  - Fix: Add YAML/JSON schema validation
  - File: shared/utils/config.mojo:from_yaml(), from_json()
  - Severity: CRITICAL

### Benchmarks Module

- [ ] **Proper JSON parsing**
  - Issue: Manual string parsing error-prone
  - Fix: Use stdlib JSON or Python interop
  - File: benchmarks/scripts/compare_results.mojo
  - Severity: CRITICAL
- [ ] **Exit code handling**
  - Issue: Could exit success despite failures
  - Fix: Add proper error handling with exit codes
  - File: benchmarks/scripts/compare_results.mojo
  - Severity: CRITICAL

## Priority 2: Major Issues (Address in Phase 3)

See CONSOLIDATED_REVIEW.md for full list.

## Assignment Strategy

- **Memory/Safety fixes**: senior-implementation-engineer
- **I/O fixes**: implementation-engineer
- **Config fixes**: implementation-engineer
- **Benchmark fixes**: performance-engineer
