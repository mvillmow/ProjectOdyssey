# Tools Catalog

Comprehensive catalog of all development tools in the ML Odyssey repository.

## Quick Reference

| Tool | Language | Purpose | Status |
|------|----------|---------|--------|
| `paper-scaffold/scaffold.py` | Python | Generate paper structure | Planned |
| `test-utils/data_generators.mojo` | Mojo | Generate test data | Planned |
| `test-utils/fixtures.mojo` | Mojo | Reusable test fixtures | Planned |
| `test-utils/coverage.py` | Python | Coverage analysis | Planned |
| `test-utils/performance.mojo` | Mojo | Performance testing | Planned |
| `benchmarking/model_bench.mojo` | Mojo | Model benchmarks | Planned |
| `benchmarking/training_bench.mojo` | Mojo | Training benchmarks | Planned |
| `benchmarking/inference_bench.mojo` | Mojo | Inference benchmarks | Planned |
| `benchmarking/memory_tracker.mojo` | Mojo | Memory usage tracking | Planned |
| `benchmarking/report_generator.py` | Python | Benchmark reports | Planned |
| `codegen/mojo_boilerplate.py` | Python | Mojo code generation | Planned |
| `codegen/training_template.py` | Python | Training loop templates | Planned |
| `codegen/data_pipeline.py` | Python | Data pipeline code | Planned |
| `codegen/metrics_generator.py` | Python | Metrics code generation | Planned |

## Paper Scaffolding

### scaffold.py

**Purpose**: Generate complete directory structure for new paper implementations

**Language**: Python (regex-heavy templating)

**Usage**:
```bash
python tools/paper-scaffold/scaffold.py \
    --paper "AlexNet" \
    --author "Krizhevsky et al." \
    --year 2012 \
    --output papers/alexnet/
```

**Generated Structure**:
```text
papers/alexnet/
├── README.md
├── model.mojo
├── train.mojo
├── test.mojo
├── data/
├── configs/
└── tests/
```

**Options**:
- `--paper NAME` - Paper name (required)
- `--author AUTHOR` - Paper author(s) (optional)
- `--year YEAR` - Publication year (optional)
- `--output DIR` - Output directory (required)
- `--template TYPE` - Template type (default: standard)

**Status**: Planned (Issue #69)

**Dependencies**: Python 3.8+, Jinja2 (for templates)

**Documentation**: [paper-scaffold/README.md](paper-scaffold/README.md)

---

## Testing Utilities

### data_generators.mojo

**Purpose**: Generate synthetic test data for ML models

**Language**: Mojo (performance-critical, type-safe)

**Usage**:
```mojo
from tools.test_utils import generate_batch, generate_image

fn test_model():
    // Generate random batch
    let batch = generate_batch(shape=(32, 3, 28, 28))
    
    // Generate specific pattern
    let zeros = generate_image(shape=(1, 28, 28), fill=0.0)
    let ones = generate_image(shape=(1, 28, 28), fill=1.0)
```

**Functions**:
- `generate_batch(shape, dtype, range)` - Random batches
- `generate_image(shape, fill, pattern)` - Synthetic images
- `generate_sequence(length, vocab_size)` - Text sequences
- `generate_labels(num_samples, num_classes)` - Classification labels

**Status**: Planned (Issue #69)

**Documentation**: [test-utils/README.md](test-utils/README.md)

### fixtures.mojo

**Purpose**: Reusable test fixtures for common model types

**Language**: Mojo (type-safe, consistent)

**Usage**:
```mojo
from tools.test_utils import ModelFixture, DataFixture

fn test_forward_pass():
    let model = ModelFixture.small_cnn()
    let data = DataFixture.mnist_batch()
    let output = model.forward(data)
```

**Fixtures**:
- `ModelFixture.small_cnn()` - Tiny CNN for testing
- `ModelFixture.linear_network()` - Simple MLP
- `DataFixture.mnist_batch()` - MNIST-like data
- `DataFixture.cifar_batch()` - CIFAR-like data

**Status**: Planned (Issue #69)

### coverage.py

**Purpose**: Test coverage analysis and reporting

**Language**: Python (integrates with pytest-cov)

**Usage**:
```bash
python tools/test-utils/coverage.py \
    --source papers/lenet5/ \
    --report-dir coverage/
```

**Features**:
- Line coverage measurement
- Branch coverage analysis
- Coverage report generation (HTML, XML)
- Diff coverage (changes only)

**Status**: Planned (Issue #69)

### performance.mojo

**Purpose**: Performance testing utilities

**Language**: Mojo (accurate timing)

**Usage**:
```mojo
from tools.test_utils import measure_latency, assert_max_latency

fn test_inference_speed():
    let model = MyModel()
    let latency = measure_latency(model, num_runs=100)
    assert_max_latency(latency, max_ms=10.0)
```

**Functions**:
- `measure_latency(fn, num_runs)` - Average latency
- `measure_throughput(fn, duration)` - Samples per second
- `assert_max_latency(latency, max_ms)` - Performance assertion
- `assert_min_throughput(throughput, min_sps)` - Throughput assertion

**Status**: Planned (Issue #69)

---

## Benchmarking

### model_bench.mojo

**Purpose**: Comprehensive model benchmarking

**Language**: Mojo (accurate ML performance measurement)

**Usage**:
```bash
mojo tools/benchmarking/model_bench.mojo \
    --paper lenet5 \
    --batch-sizes 1,8,32,128 \
    --output benchmarks/lenet5.json
```

**Measurements**:
- Forward pass latency (per batch size)
- Backward pass latency
- Memory footprint
- FLOPs count

**Output Format**: JSON
```json
{
  "model": "lenet5",
  "batch_sizes": [1, 8, 32, 128],
  "latency_ms": [0.5, 1.2, 3.1, 10.4],
  "memory_mb": [10, 25, 80, 320],
  "flops": 420000
}
```

**Status**: Planned (Issue #69)

### training_bench.mojo

**Purpose**: Training performance benchmarking

**Language**: Mojo (accurate training metrics)

**Usage**:
```bash
mojo tools/benchmarking/training_bench.mojo \
    --paper lenet5 \
    --epochs 1 \
    --batch-size 32
```

**Measurements**:
- Training throughput (samples/sec)
- Epoch duration
- Step duration
- GPU utilization
- Memory usage over time

**Status**: Planned (Issue #69)

### inference_bench.mojo

**Purpose**: Inference-specific benchmarking

**Language**: Mojo (production-like measurement)

**Usage**:
```bash
mojo tools/benchmarking/inference_bench.mojo \
    --model lenet5 \
    --batch-size 1 \
    --num-runs 1000
```

**Measurements**:
- Mean latency
- P50/P95/P99 latency
- Throughput
- Memory per request

**Status**: Planned (Issue #69)

### memory_tracker.mojo

**Purpose**: Memory usage tracking

**Language**: Mojo (direct memory access)

**Usage**:
```bash
mojo tools/benchmarking/memory_tracker.mojo \
    --paper lenet5 \
    --track training
```

**Measurements**:
- Peak memory usage
- Memory over time
- Memory by component (model, data, optimizer)
- Memory leaks detection

**Status**: Planned (Issue #69)

### report_generator.py

**Purpose**: Generate benchmark reports

**Language**: Python (matplotlib, pandas)

**Usage**:
```bash
python tools/benchmarking/report_generator.py \
    --input benchmarks/lenet5.json \
    --output benchmarks/lenet5_report.html \
    --format html
```

**Output Formats**:
- HTML (interactive charts)
- PDF (static report)
- Markdown (text report)

**Status**: Planned (Issue #69)

---

## Code Generation

### mojo_boilerplate.py

**Purpose**: Generate Mojo struct and function boilerplate

**Language**: Python (string templating)

**Usage**:
```bash
# Generate layer
python tools/codegen/mojo_boilerplate.py \
    --type layer \
    --name Conv2D \
    --params "in_channels,out_channels,kernel_size"

# Generate struct
python tools/codegen/mojo_boilerplate.py \
    --type struct \
    --name ModelConfig \
    --fields "learning_rate:Float64,batch_size:Int"
```

**Template Types**:
- `layer` - Neural network layer
- `struct` - Data structure
- `function` - Function signature
- `trait` - Trait definition

**Status**: Planned (Issue #69)

### training_template.py

**Purpose**: Generate training loop boilerplate

**Language**: Python (code generation)

**Usage**:
```bash
python tools/codegen/training_template.py \
    --optimizer SGD \
    --loss CrossEntropy \
    --metrics "accuracy,loss" \
    --output train.mojo
```

**Options**:
- `--optimizer` - Optimizer type (SGD, Adam, RMSprop)
- `--loss` - Loss function
- `--metrics` - Metrics to track
- `--scheduler` - Learning rate scheduler
- `--early-stopping` - Add early stopping

**Status**: Planned (Issue #69)

### data_pipeline.py

**Purpose**: Generate data loading and preprocessing code

**Language**: Python (code generation)

**Usage**:
```bash
python tools/codegen/data_pipeline.py \
    --dataset MNIST \
    --augmentation "flip,rotate,crop" \
    --output data.mojo
```

**Features**:
- Data loader generation
- Augmentation pipeline
- Preprocessing steps
- Batching logic

**Status**: Planned (Issue #69)

### metrics_generator.py

**Purpose**: Generate metrics calculation code

**Language**: Python (code generation)

**Usage**:
```bash
python tools/codegen/metrics_generator.py \
    --metrics "accuracy,precision,recall,f1" \
    --output metrics.mojo
```

**Metrics**:
- Classification metrics
- Regression metrics
- Custom metric templates

**Status**: Planned (Issue #69)

---

## Tool Selection Guide

### When to Use Each Tool

**Paper Scaffolding**:
- ✅ Starting new paper implementation
- ✅ Need consistent directory structure
- ❌ One-off file creation (create manually)

**Test Utilities**:
- ✅ Need synthetic test data
- ✅ Want reusable test fixtures
- ✅ Performance testing required
- ❌ Simple test cases (write directly)

**Benchmarking**:
- ✅ Performance comparison across papers
- ✅ Regression testing (detect slowdowns)
- ✅ Production deployment sizing
- ❌ Quick spot checks (use manual timing)

**Code Generation**:
- ✅ Repetitive boilerplate (layers, training loops)
- ✅ Following strict patterns
- ✅ Large-scale code updates
- ❌ Custom logic (write manually)

### Decision Tree

```text
Need to...
├── Create new paper?
│   └─→ paper-scaffold/scaffold.py
│
├── Write tests?
│   ├── Need test data? → test-utils/data_generators.mojo
│   ├── Need fixtures? → test-utils/fixtures.mojo
│   └── Need performance checks? → test-utils/performance.mojo
│
├── Measure performance?
│   ├── Model inference? → benchmarking/model_bench.mojo
│   ├── Training speed? → benchmarking/training_bench.mojo
│   └── Memory usage? → benchmarking/memory_tracker.mojo
│
└── Generate code?
    ├── Structs/layers? → codegen/mojo_boilerplate.py
    ├── Training loop? → codegen/training_template.py
    ├── Data pipeline? → codegen/data_pipeline.py
    └── Metrics code? → codegen/metrics_generator.py
```

---

## Installation

See [INSTALL.md](./INSTALL.md) for setup instructions.

Quick start:
```bash
python tools/setup/install_tools.py
python tools/setup/verify_tools.py
```

---

## Contributing

### Adding a New Tool

1. Identify category (scaffolding, testing, benchmarking, codegen)
2. Choose language (follow ADR-001)
3. Create tool in appropriate directory
4. Write documentation (README with examples)
5. Add to this catalog
6. Update INTEGRATION.md

### Tool Documentation Template

```markdown
# Tool Name

## Purpose
One-sentence description

## Usage
```bash
command example
```

## Options
- `--option` - Description

## Examples
Practical examples

## Status
Planned/In Progress/Complete
```

---

## References

- [INTEGRATION.md](./INTEGRATION.md) - Integration guide
- [INSTALL.md](./INSTALL.md) - Installation instructions
- [ADR-001](../notes/review/adr/ADR-001-language-selection-tooling.md) - Language selection
- [Tools README](./README.md) - Overview

---

**Document**: `/tools/CATALOG.md`
**Last Updated**: 2025-11-16
**Status**: Living document
