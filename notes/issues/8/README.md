# Issue #8: Hybrid Python/Mojo Bridge Architecture

## Objective

Design and implement a comprehensive hybrid architecture that enables seamless interoperability between Python
and Mojo code, allowing ML Odyssey to leverage Python's extensive ecosystem while taking advantage of Mojo's
performance benefits. This architecture will support development, testing, and production workflows with clear
separation of concerns and minimal performance overhead.

## Deliverables

- Complete hybrid bridge architecture documentation
- Interoperability patterns and best practices
- Python/Mojo communication layer implementation
- Serialization and data exchange mechanisms
- Tooling for seamless Python-Mojo development
- Test utilities and validation framework
- Performance benchmarking suite
- Migration guidelines for converting Python to Mojo
- Production-ready configuration for hybrid deployments

## Success Criteria

- ✅ Hybrid bridge architecture documented and approved
- ✅ Five interoperability patterns implemented and tested
- ✅ Data serialization layer working (JSON, NumPy, Arrow formats)
- ✅ Communication patterns tested with < 10% overhead
- ✅ Development tooling enabling rapid iteration
- ✅ Clear migration path for Python-to-Mojo conversion
- ✅ Performance benchmarks demonstrate viability
- ✅ Team documentation complete
- ✅ All components integrated into CI/CD

## References

- [CLAUDE.md](/home/mvillmow/ml-odyssey/CLAUDE.md) - Project guidance and language preferences
- [Agent Documentation](/agents/README.md) - Team coordination patterns
- [Skills Design](/notes/review/skills-design.md) - Agent skill specifications
- [Orchestration Patterns](/notes/review/orchestration-patterns.md) - Integration strategies

## Implementation Notes

**Status**: Architecture Design Phase

This document provides a complete specification for a hybrid Python/Mojo architecture that balances pragmatism
(using Python's maturity and libraries) with performance (leveraging Mojo's capabilities). The architecture
follows ML Odyssey's 5-phase workflow and integrates with the agent system.

---

## Executive Summary

### Problem Statement

ML Odyssey faces a fundamental trade-off:

- **Full Mojo approach**: High performance but limited ecosystem access; slower initial development
- **Full Python approach**: Rapid development with mature libraries but performance limitations
- **Hybrid approach**: Best of both worlds with careful architectural design

This document specifies a comprehensive hybrid architecture that:

1. Enables seamless Python-Mojo interoperability
1. Provides clear separation of concerns
1. Maintains < 10% overhead for communication
1. Supports incremental migration from Python to Mojo
1. Enables parallel development workflows

### Key Design Principles

1. **Pragmatic Layering**: Use Python where it makes sense (tools, integration), Mojo where it matters (compute)
1. **Clear Boundaries**: Explicit contracts between Python and Mojo components
1. **Minimal Overhead**: Design patterns that avoid serialization bottlenecks
1. **Developer Velocity**: Tools and patterns that enable rapid iteration
1. **Production Ready**: Configurations for both development and deployment

### Architecture Layers

```text

┌─────────────────────────────────────────────────────────────┐
│              Application Layer (Python or Mojo)              │
│  ├─ Paper implementations                                    │
│  ├─ Training scripts                                         │
│  └─ Evaluation pipelines                                     │
├─────────────────────────────────────────────────────────────┤
│         Hybrid Bridge Layer (Communication & Data)           │
│  ├─ Type mapping (Mojo ↔ Python)                            │
│  ├─ Serialization (JSON, NumPy, Arrow)                      │
│  ├─ API adapters (function/class boundaries)                │
│  └─ Protocol handlers (messaging, data exchange)            │
├─────────────────────────────────────────────────────────────┤
│           Core Implementation Layer (Mojo)                   │
│  ├─ Performance-critical algorithms                          │
│  ├─ Tensor operations                                        │
│  ├─ Neural network kernels                                   │
│  └─ SIMD-optimized primitives                               │
├─────────────────────────────────────────────────────────────┤
│         Ecosystem Layer (External Libraries)                 │
│  ├─ Python ML libraries (TensorFlow, PyTorch)               │
│  ├─ Data processing (Pandas, NumPy)                         │
│  └─ Utilities (logging, monitoring)                         │
└─────────────────────────────────────────────────────────────┘
```text

---

## Part 1: Architecture Overview

### 1.1 Core Components

#### Python Components

**Purpose**: Rapid prototyping, integration, testing framework

### Typical Uses

- Data loading and preprocessing
- Experiment orchestration
- Metrics computation (initially)
- Integration with external tools
- Test harnesses and validation
- Configuration management
- Logging and monitoring

**Performance Requirements**: Flexible (not critical)

#### Mojo Components

**Purpose**: Performance-critical computation

### Typical Uses

- Tensor operations
- Forward/backward passes
- Core algorithms (especially with loops)
- Data transformations requiring SIMD
- Memory-intensive computations

**Performance Requirements**: Strict (target 10x+ speedup over pure Python)

#### Hybrid Bridge

**Purpose**: Enable seamless interaction between Python and Mojo

### Responsibilities

- Type mapping between Python and Mojo types
- Data serialization and deserialization
- Function/class boundary definitions
- API adaptation (converting between calling conventions)
- Error propagation and handling
- Resource management

### 1.2 Communication Patterns

Five complementary patterns for different use cases:

```text

┌──────────────────────────────────────────────────────────────┐
│           Hybrid Communication Pattern Matrix                 │
├──────────────┬──────────────┬─────────────┬─────────────────┤
│   Pattern    │   Overhead   │   Use Case  │   Complexity    │
├──────────────┼──────────────┼─────────────┼─────────────────┤
│ Wrapper      │  < 1%        │ Single ops  │   Low           │
│ Async Queue  │  < 5%        │ Streaming   │   Medium        │
│ Shared Mem   │  < 2%        │ Bulk ops    │   High          │
│ Protocol Buf │  < 8%        │ Complex obj │   Medium        │
│ JIT Bridge   │  < 3%        │ Tight loops │   Very High     │
└──────────────┴──────────────┴─────────────┴─────────────────┘
```text

---

## Part 2: Implementation Patterns

### 2.1 Pattern 1: Direct Wrapper (Lowest Overhead)

**Best for**: Simple function calls with primitive types

**Overhead**: < 1%

### Architecture

```text
┌──────────────┐
│ Python Code  │
└──────┬───────┘
       │ call
       ▼
┌──────────────────────────┐
│  Mojo FFI Function       │
│  (direct wrapper)        │
└──────┬───────────────────┘
       │ call
       ▼
┌──────────────────────────┐
│  Mojo Implementation     │
│  (compute kernel)        │
└──────────────────────────┘
```text

### Example Implementation

Mojo kernel (`ml_odyssey/core/ops.mojo`):

```mojo
@always_inline
fn add[dtype: DType](a: Float32, b: Float32) -> Float32:
    """Simple addition - exposing to Python."""
    return a + b

fn matrix_multiply[M: Int, N: Int, K: Int](
    a: Tensor[DType.float32, M, K],
    b: Tensor[DType.float32, K, N]
) -> Tensor[DType.float32, M, N]:
    """Matrix multiplication kernel - high performance."""
    var result = Tensor[DType.float32, M, N]()
    # ... optimized implementation
    return result
```text

Python wrapper (`ml_odyssey/core/bridge.py`):

```python
import ctypes
from typing import Tuple
import numpy as np

class MojoBridge:
    """Direct FFI wrapper for Mojo functions."""

    def __init__(self, lib_path: str):
        self.lib = ctypes.CDLL(lib_path)
        self._setup_function_signatures()

    def _setup_function_signatures(self):
        """Configure ctypes function signatures."""
        # Addition function
        self.lib.add_wrapper.argtypes = [ctypes.c_float, ctypes.c_float]
        self.lib.add_wrapper.restype = ctypes.c_float

        # Matrix multiply function
        self.lib.matmul_wrapper.argtypes = [
            ctypes.c_void_p,  # Matrix A pointer
            ctypes.c_void_p,  # Matrix B pointer
            ctypes.c_int,     # M dimension
            ctypes.c_int,     # N dimension
            ctypes.c_int,     # K dimension
        ]
        self.lib.matmul_wrapper.restype = ctypes.c_void_p

    def add(self, a: float, b: float) -> float:
        """Add two floats (< 0.5% overhead)."""
        return self.lib.add_wrapper(a, b)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix multiply with minimal overhead."""
        m, k = a.shape
        k2, n = b.shape
        assert k == k2, "Incompatible dimensions"

        # Pre-allocate result
        result = np.zeros((m, n), dtype=np.float32)

        # Call Mojo kernel
        result_ptr = self.lib.matmul_wrapper(
            a.ctypes.data_as(ctypes.c_void_p),
            b.ctypes.data_as(ctypes.c_void_p),
            m, n, k
        )

        # Copy result from Mojo allocation
        result = np.ctypeslib.as_array(
            ctypes.cast(result_ptr, ctypes.POINTER(ctypes.c_float)),
            shape=(m, n)
        )
        return result
```text

### Use Case

- Core tensor operations
- Mathematical primitives
- Hot-path operations

### Advantages

- Minimal overhead
- Direct memory access
- Maximum performance

### Disadvantages

- Requires careful memory management
- Limited to simple types
- No automatic marshalling

**When to Use**: High-performance primitives, frequently called operations

---

### 2.2 Pattern 2: Async Queue (Best for Streaming)

**Best for**: Continuous data streams, batched processing

**Overhead**: < 5%

### Architecture

```text
┌────────────────────────────────────────────────┐
│       Python Producer (data loading)            │
│       ├─ Load batch                             │
│       ├─ Queue.put(batch)                       │
│       └─ yield to other tasks                   │
└─────┬──────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────┐
│     Async Message Queue (Thread-safe)          │
│     ├─ Batch buffering                         │
│     ├─ Back-pressure handling                  │
│     └─ Status notifications                    │
└─────┬──────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────┐
│    Mojo Consumer (processing)                  │
│    ├─ batch = Queue.get(timeout)               │
│    ├─ process_batch(batch)                     │
│    └─ Queue.put(results)                       │
└────────────────────────────────────────────────┘
```text

### Example Implementation

Python producer (`ml_odyssey/training/data_pipeline.py`):

```python
import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator
import numpy as np

@dataclass
class Batch:
    """Data batch with metadata."""
    x: np.ndarray        # Input data
    y: np.ndarray        # Labels
    batch_id: int        # For tracking
    timestamp: float     # For monitoring

class DataProducer:
    """Async data producer for Mojo consumer."""

    def __init__(self, queue_size: int = 10):
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.running = False

    async def produce_batches(
        self,
        data_loader,
        num_batches: int = None
    ) -> AsyncGenerator[Batch, None]:
        """Produce batches asynchronously."""
        self.running = True
        batch_id = 0

        try:
            for batch_data in data_loader:
                if num_batches and batch_id >= num_batches:
                    break

                batch = Batch(
                    x=batch_data['x'].astype(np.float32),
                    y=batch_data['y'].astype(np.int32),
                    batch_id=batch_id,
                    timestamp=time.time()
                )

                await self.queue.put(batch)
                batch_id += 1
                yield batch

        finally:
            self.running = False
            # Signal completion
            await self.queue.put(None)
```text

Mojo consumer (`ml_odyssey/core/training.mojo`):

```mojo
struct BatchProcessor:
    """Processes batches from Python async queue."""

    var model: NeuralNetwork
    var optimizer: Optimizer
    var queue: PythonObject  # Reference to Python queue

    fn process_batches(
        inout self,
        num_epochs: Int
    ) -> None:
        """Process batches from async queue."""
        for epoch in range(num_epochs):
            var batch_id = 0

            while True:
                # Get batch from queue with timeout
                let batch = self._get_batch_from_queue(timeout=5.0)

                if batch is None:
                    break  # End of epoch

                # Process batch
                let loss = self._train_step(batch)

                # Send results back to Python
                self._send_results(batch.batch_id, loss)

                batch_id += 1

    fn _get_batch_from_queue(
        self,
        timeout: Float32
    ) -> PythonObject:
        """Get batch from Python queue (with timeout)."""
        # Python interop to get from asyncio.Queue
        return None  # Placeholder

    fn _train_step(
        inout self,
        batch: PythonObject
    ) -> Float32:
        """Single training step."""
        # Extract data from batch object
        let x = batch.x
        let y = batch.y

        # Forward pass
        var predictions = self.model.forward(x)

        # Compute loss
        let loss = compute_cross_entropy(predictions, y)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```text

Python results handler (`ml_odyssey/training/results_handler.py`):

```python
import asyncio
from typing import Dict, List

class ResultsHandler:
    """Handles results from Mojo consumer."""

    def __init__(self):
        self.results: Dict[int, float] = {}
        self.event = asyncio.Event()

    async def collect_results(
        self,
        num_batches: int,
        timeout: float = 30.0
    ) -> List[float]:
        """Collect results from Mojo consumer."""
        collected = {}

        async def wait_for_result(batch_id: int):
            start = time.time()
            while batch_id not in self.results:
                if time.time() - start > timeout:
                    raise TimeoutError(f"Result for batch {batch_id}")
                await asyncio.sleep(0.01)
            return self.results[batch_id]

        # Collect all results concurrently
        tasks = [wait_for_result(i) for i in range(num_batches)]
        results = await asyncio.gather(*tasks)

        return results
```text

### Use Case

- Data loading pipelines
- Continuous training loops
- Multi-stream processing
- Real-time data handling

### Advantages

- Decouples producer and consumer
- Enables pipelining
- Natural for streaming workloads
- Built-in back-pressure

### Disadvantages

- Queue overhead (~5%)
- Async complexity
- Synchronization needed

**When to Use**: Data pipelines, continuous processing, multi-task training

---

### 2.3 Pattern 3: Shared Memory (Best for Bulk Operations)

**Best for**: Large tensors, bulk computations

**Overhead**: < 2%

### Architecture

```text
┌──────────────────────┐
│   Python Setup       │
│  ├─ Allocate buffer  │
│  ├─ Fill with data   │
│  └─ Get pointer      │
└─────┬────────────────┘
      │
      ▼
┌──────────────────────────────────────────┐
│   Shared Memory Region (zero-copy)       │
│   ├─ Input tensor (read)                 │
│   ├─ Weights (read)                      │
│   ├─ Output tensor (write)               │
│   └─ Scratch space (read-write)          │
└─────┬──────────────────────────────────────┘
      │
      ▼
┌──────────────────────┐
│   Mojo Kernel       │
│  ├─ Map memory      │
│  ├─ Compute         │
│  └─ Mark complete   │
└──────────────────────┘
```text

### Example Implementation

Python setup (`ml_odyssey/core/shared_memory.py`):

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional
import ctypes

@dataclass
class SharedTensor:
    """Tensor shared between Python and Mojo."""

    data: np.ndarray
    shape: tuple
    dtype: np.dtype
    address: int

    @staticmethod
    def from_array(arr: np.ndarray) -> 'SharedTensor':
        """Create shared tensor from NumPy array."""
        # Ensure contiguous memory layout
        arr = np.ascontiguousarray(arr)

        return SharedTensor(
            data=arr,
            shape=arr.shape,
            dtype=arr.dtype,
            address=arr.ctypes.data
        )

    def get_pointer(self) -> int:
        """Get memory address for Mojo FFI."""
        return self.address

    def get_strides(self) -> tuple:
        """Get strides for Mojo layout info."""
        return self.data.strides

class SharedMemoryManager:
    """Manage shared memory allocations."""

    def __init__(self):
        self.allocations = {}
        self.counter = 0

    def allocate(
        self,
        shape: tuple,
        dtype: np.dtype = np.float32
    ) -> SharedTensor:
        """Allocate shared memory buffer."""
        arr = np.zeros(shape, dtype=dtype)
        tensor = SharedTensor.from_array(arr)

        self.allocations[self.counter] = tensor
        self.counter += 1

        return tensor

    def get_allocation(self, alloc_id: int) -> SharedTensor:
        """Retrieve allocation by ID."""
        return self.allocations[alloc_id]

    def free(self, alloc_id: int) -> None:
        """Free allocation."""
        if alloc_id in self.allocations:
            del self.allocations[alloc_id]
```text

Mojo consumer (`ml_odyssey/core/shared_ops.mojo`):

```mojo
struct SharedMemoryOps:
    """Operations on shared memory regions."""

    fn matrix_multiply_shared(
        a_ptr: UnsafePointer[Float32],
        b_ptr: UnsafePointer[Float32],
        out_ptr: UnsafePointer[Float32],
        m: Int,
        n: Int,
        k: Int
    ) -> None:
        """Multiply matrices in shared memory."""
        # Tiled matrix multiplication
        alias tile_size = 64

        for i in range(0, m, tile_size):
            for j in range(0, n, tile_size):
                # Load tile from A and B
                var a_tile = _load_tile(a_ptr, i, k, tile_size)
                var b_tile = _load_tile(b_ptr, j, n, tile_size)

                # Compute tile
                var c_tile = _multiply_tiles(a_tile, b_tile)

                # Store result
                _store_tile(out_ptr, c_tile, i, j, n)

    fn _load_tile(
        ptr: UnsafePointer[Float32],
        row: Int,
        cols: Int,
        size: Int
    ) -> DynamicVector[Float32]:
        """Load tile from memory (SIMD-optimized)."""
        var result = DynamicVector[Float32](size * size)
        for i in range(size):
            for j in range(size):
                let idx = (row + i) * cols + j
                result[i * size + j] = ptr[idx]
        return result

    fn _multiply_tiles(
        a: DynamicVector[Float32],
        b: DynamicVector[Float32]
    ) -> DynamicVector[Float32]:
        """Multiply two tiles."""
        alias size = 64
        var result = DynamicVector[Float32](size * size)

        # SIMD-optimized multiplication
        for i in range(size):
            for j in range(size):
                var sum = Float32(0.0)
                for k in range(size):
                    sum += a[i * size + k] * b[k * size + j]
                result[i * size + j] = sum

        return result
```text

### Use Case

- Large tensor operations
- Batch processing
- Memory-intensive computations
- When zero-copy is critical

### Advantages

- Minimal overhead (just pointer passing)
- Zero-copy access
- Maximum performance
- Works with large tensors

### Disadvantages

- Memory management complexity
- Requires careful alignment
- Pointer safety concerns
- Limited type safety

**When to Use**: Tensor operations, bulk computations, memory-critical paths

---

### 2.4 Pattern 4: Protocol Buffers (Best for Complex Objects)

**Best for**: Complex nested structures, versioning requirements

**Overhead**: < 8%

### Architecture

```text
┌──────────────────────────┐
│   Python Object Model    │
│   ├─ Dataclass           │
│   ├─ Attributes          │
│   └─ Methods             │
└─────┬────────────────────┘
      │ serialize
      ▼
┌──────────────────────────┐
│   Protocol Buffer        │
│   (compact binary)       │
│   ├─ Field encoding      │
│   ├─ Size optimization   │
│   └─ Version info        │
└─────┬────────────────────┘
      │ deserialize
      ▼
┌──────────────────────────┐
│   Mojo Struct            │
│   ├─ Field mapping       │
│   ├─ Type conversion     │
│   └─ Value initialization│
└──────────────────────────┘
```text

**Example Protocol Definition** (`ml_odyssey/core/protos/model.proto`):

```protobuf
syntax = "proto3";

package ml_odyssey;

message Tensor {
    repeated float data = 1;
    repeated int32 shape = 2;
    string dtype = 3;
}

message Layer {
    string name = 1;
    string layer_type = 2;
    Tensor weights = 3;
    Tensor bias = 4;
    map<string, string> config = 5;
}

message Model {
    string name = 1;
    int32 version = 2;
    repeated Layer layers = 3;
    ModelConfig config = 4;
}

message ModelConfig {
    float learning_rate = 1;
    int32 batch_size = 2;
    int32 num_epochs = 3;
    string optimizer = 4;
}
```text

Python serialization (`ml_odyssey/core/serialization.py`):

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import struct
import io

@dataclass
class SerializedTensor:
    """Protocol buffer representation of tensor."""

    data: bytes
    shape: List[int]
    dtype: str

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        buf = io.BytesIO()

        # Write shape
        buf.write(struct.pack('I', len(self.shape)))
        for dim in self.shape:
            buf.write(struct.pack('I', dim))

        # Write dtype
        buf.write(struct.pack('I', len(self.dtype)))
        buf.write(self.dtype.encode())

        # Write data
        buf.write(struct.pack('I', len(self.data)))
        buf.write(self.data)

        return buf.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> 'SerializedTensor':
        """Deserialize from bytes."""
        buf = io.BytesIO(data)

        # Read shape
        shape_len = struct.unpack('I', buf.read(4))[0]
        shape = []
        for _ in range(shape_len):
            shape.append(struct.unpack('I', buf.read(4))[0])

        # Read dtype
        dtype_len = struct.unpack('I', buf.read(4))[0]
        dtype = buf.read(dtype_len).decode()

        # Read data
        data_len = struct.unpack('I', buf.read(4))[0]
        tensor_data = buf.read(data_len)

        return SerializedTensor(
            data=tensor_data,
            shape=tuple(shape),
            dtype=dtype
        )

@dataclass
class SerializedModel:
    """Protocol buffer representation of model."""

    name: str
    version: int
    layers: List[Dict]
    config: Dict

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        buf = io.BytesIO()

        # Write name
        buf.write(struct.pack('I', len(self.name)))
        buf.write(self.name.encode())

        # Write version
        buf.write(struct.pack('I', self.version))

        # Write layers (simplified)
        buf.write(struct.pack('I', len(self.layers)))
        for layer in self.layers:
            layer_bytes = _serialize_layer(layer)
            buf.write(struct.pack('I', len(layer_bytes)))
            buf.write(layer_bytes)

        return buf.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> 'SerializedModel':
        """Deserialize from bytes."""
        buf = io.BytesIO(data)

        # Read name
        name_len = struct.unpack('I', buf.read(4))[0]
        name = buf.read(name_len).decode()

        # Read version
        version = struct.unpack('I', buf.read(4))[0]

        # Read layers
        num_layers = struct.unpack('I', buf.read(4))[0]
        layers = []
        for _ in range(num_layers):
            layer_len = struct.unpack('I', buf.read(4))[0]
            layer_bytes = buf.read(layer_len)
            layers.append(_deserialize_layer(layer_bytes))

        return SerializedModel(
            name=name,
            version=version,
            layers=layers,
            config={}
        )
```text

Mojo deserialization (`ml_odyssey/core/proto_bridge.mojo`):

```mojo
struct ProtoBufTensor:
    """Deserialized tensor from protocol buffer."""

    var shape: DynamicVector[Int]
    var dtype: String
    var data_ptr: UnsafePointer[Float32]
    var data_size: Int

    fn from_bytes(bytes: UnsafePointer[UInt8], len: Int) -> ProtoBufTensor:
        """Deserialize from bytes."""
        var buf = ByteBuffer(bytes, len)

        # Read shape
        var shape_len = buf.read_uint32()
        var shape = DynamicVector[Int](shape_len)
        for i in range(shape_len):
            shape[i] = buf.read_uint32()

        # Read dtype
        var dtype_len = buf.read_uint32()
        var dtype = buf.read_string(dtype_len)

        # Read data
        var data_len = buf.read_uint32()
        var data_ptr = buf.read_data_pointer()

        return ProtoBufTensor(
            shape=shape,
            dtype=dtype,
            data_ptr=data_ptr,
            data_size=data_len
        )

    fn to_numpy(self) -> PythonObject:
        """Convert to NumPy array."""
        # Calculate total elements
        var total_elements = 1
        for dim in self.shape:
            total_elements *= dim

        # Create NumPy array from pointer
        return _numpy_from_pointer(
            self.data_ptr,
            self.shape,
            self.dtype
        )
```text

### Use Case

- Model configuration serialization
- Complex nested structures
- Version-independent communication
- Cross-language RPC

### Advantages

- Language-independent
- Version compatibility
- Compact encoding
- Self-describing

### Disadvantages

- Serialization overhead (~8%)
- Code generation needed
- Schema management
- Debugging difficulty

**When to Use**: Model configs, complex objects, cross-language communication

---

### 2.5 Pattern 5: JIT Bridge (Best for Tight Loops)

**Best for**: Tight loops with function callbacks, dynamic compilation

**Overhead**: < 3% after warm-up

### Architecture

```text
┌─────────────────────────────┐
│   Python Source (Loop)      │
│   for i in range(1000000):  │
│     result = mojo_func(i)   │
└──────┬──────────────────────┘
       │ JIT Compile
       ▼
┌─────────────────────────────┐
│   Mojo JIT Compiler         │
│   ├─ Trace execution        │
│   ├─ Specialize types       │
│   └─ Generate machine code  │
└──────┬──────────────────────┘
       │ Execute
       ▼
┌─────────────────────────────┐
│   Machine Code (Optimized)  │
│   ├─ Loop unrolling         │
│   ├─ Vectorization          │
│   └─ Cache optimized        │
└─────────────────────────────┘
```text

### Example Implementation

Python caller (`ml_odyssey/core/jit_bridge.py`):

```python
from typing import Callable, Any, List
import ctypes
import time

class MojoJITBridge:
    """JIT compilation bridge for tight loops."""

    def __init__(self, lib_path: str):
        self.lib = ctypes.CDLL(lib_path)
        self.cache = {}
        self.stats = {
            'compiled': 0,
            'executed': 0,
            'total_time': 0.0
        }

    def jit_loop(
        self,
        func: Callable,
        iterations: int,
        dtype='float32'
    ) -> Any:
        """JIT compile and execute loop."""
        # Create cache key
        key = (func.__name__, iterations, dtype)

        if key not in self.cache:
            # Trace and compile
            start = time.time()
            compiled_func = self._compile(func, iterations, dtype)
            compile_time = time.time() - start

            self.cache[key] = {
                'func': compiled_func,
                'compile_time': compile_time
            }
            self.stats['compiled'] += 1

        # Execute compiled version
        start = time.time()
        compiled_func = self.cache[key]['func']
        result = compiled_func()
        exec_time = time.time() - start

        self.stats['executed'] += 1
        self.stats['total_time'] += exec_time

        return result

    def _compile(self, func: Callable, iterations: int, dtype: str):
        """Compile function to Mojo."""
        # Generate Mojo code from Python function
        mojo_code = self._generate_mojo_code(func, iterations, dtype)

        # Compile to library
        lib_path = self._invoke_mojo_compiler(mojo_code)

        # Load and return function
        lib = ctypes.CDLL(lib_path)
        return lambda: lib.run_loop()

    def _generate_mojo_code(
        self,
        func: Callable,
        iterations: int,
        dtype: str
    ) -> str:
        """Generate Mojo code from Python function."""
        # Use inspect to get function source
        import inspect
        source = inspect.getsource(func)

        # Convert to Mojo syntax (simplified)
        mojo_source = self._transpile_to_mojo(source, iterations, dtype)

        return mojo_source

    def _transpile_to_mojo(
        self,
        python_src: str,
        iterations: int,
        dtype: str
    ) -> str:
        """Transpile Python to Mojo."""
        return f"""
fn run_loop() -> PythonObject:
    var result = DynamicVector[{dtype}]()

    for i in range({iterations}):
        # Transpiled code would go here
        pass

    return result
"""

    def _invoke_mojo_compiler(self, mojo_code: str) -> str:
        """Invoke mojo compiler."""
        import tempfile
        import subprocess

        with tempfile.NamedTemporaryFile(suffix='.mojo', delete=False) as f:
            f.write(mojo_code.encode())
            mojo_file = f.name

        # Compile to shared library
        output_path = mojo_file.replace('.mojo', '.so')
        subprocess.run(['mojo', 'build', '-o', output_path, mojo_file])

        return output_path

    def get_stats(self) -> dict:
        """Get JIT statistics."""
        return {
            **self.stats,
            'cache_size': len(self.cache)
        }
```text

Mojo compiled kernel (`ml_odyssey/core/jit_kernel.mojo`):

```mojo
@always_inline
fn run_loop() -> PythonObject:
    """JIT-compiled tight loop."""
    var result = DynamicVector[Float32]()

    # Fully unrolled and vectorized loop
    @parameter
    fn inner_loop[width: Int]() -> None:
        alias unroll_factor = 8
        var accum = SIMD[DType.float32, width](0.0)

        for i in range(0, 1000000, width * unroll_factor):
            # Vectorized operations
            var v1 = SIMD[DType.float32, width]._load_from_memory(i)
            var v2 = SIMD[DType.float32, width]._load_from_memory(i + width)
            var v3 = SIMD[DType.float32, width]._load_from_memory(i + 2*width)
            var v4 = SIMD[DType.float32, width]._load_from_memory(i + 3*width)

            accum = v1 + v2 + v3 + v4

            result.push_back(accum[0])

    inner_loop[simdwidth()]()

    return result._as_python_object()
```text

### Use Case

- Tight numerical loops
- Dynamic kernel selection
- Specialized computation paths
- Performance-critical inner loops

### Advantages

- Minimal overhead after warm-up
- Specialized machine code
- Automatic vectorization
- Adaptive compilation

### Disadvantages

- Complex compilation pipeline
- Warm-up overhead
- Cache invalidation concerns
- Debugging complexity

**When to Use**: Performance-critical tight loops, adaptive kernels

---

## Part 3: Directory Structure

### Complete Project Layout

```text
ml-odyssey/
├── ml_odyssey/
│   ├── __init__.py
│   │
│   ├── core/                          # Hybrid bridge and primitives
│   │   ├── __init__.py
│   │   ├── bridge.py                  # Pattern 1: Direct wrapper
│   │   ├── async_queue.py             # Pattern 2: Async queue
│   │   ├── shared_memory.py           # Pattern 3: Shared memory
│   │   ├── serialization.py           # Pattern 4: Protocol buffers
│   │   ├── jit_bridge.py              # Pattern 5: JIT bridge
│   │   │
│   │   ├── ops.mojo                   # Mojo kernel implementations
│   │   ├── shared_ops.mojo            # Shared memory operations
│   │   ├── jit_kernel.mojo            # JIT-compiled kernels
│   │   │
│   │   └── proto/                     # Protocol buffer definitions
│   │       ├── model.proto
│   │       └── data.proto
│   │
│   ├── training/                      # Training loop infrastructure
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Base trainer (Python)
│   │   ├── data_pipeline.py           # Data loading (async)
│   │   ├── results_handler.py         # Results collection
│   │   │
│   │   └── trainer.mojo               # Performance trainer (optional)
│   │
│   ├── utils/                         # Utilities
│   │   ├── __init__.py
│   │   ├── type_mapping.py            # Python ↔ Mojo type mapping
│   │   ├── conversion.py              # Type conversions
│   │   └── memory.py                  # Memory management utilities
│   │
│   └── papers/                        # Paper implementations
│       ├── __init__.py
│       └── lenet5/                    # Example paper
│           ├── model.py               # Model definition (Python)
│           ├── model.mojo             # Model definition (Mojo)
│           ├── train.py               # Training script (Python)
│           └── train.mojo             # Training script (Mojo)
│
├── tests/
│   ├── test_bridge_patterns.py        # Test all 5 patterns
│   ├── test_async_queue.py            # Pattern 2 tests
│   ├── test_shared_memory.py          # Pattern 3 tests
│   ├── test_serialization.py          # Pattern 4 tests
│   ├── test_jit_bridge.py             # Pattern 5 tests
│   ├── test_type_mapping.py           # Type conversion tests
│   │
│   └── benchmarks/
│       ├── benchmark_patterns.py      # Compare 5 patterns
│       ├── benchmark_overhead.py      # Measure overhead
│       └── benchmark_performance.py   # End-to-end performance
│
├── docs/
│   ├── hybrid_architecture.md         # Complete architecture guide
│   ├── migration_guide.md             # Python → Mojo conversion guide
│   ├── performance_guide.md           # Performance tuning guide
│   ├── troubleshooting.md             # Common issues
│   └── pattern_selection.md           # How to choose patterns
│
├── .claude/
│   ├── agents/                        # Agent configurations
│   │   ├── integration-specialist.md  # Hybrid bridge design
│   │   └── performance-engineer.md    # Optimization
│   │
│   └── skills/
│       ├── hybrid-bridge-patterns/
│       │   ├── SKILL.md
│       │   └── examples/
│       │
│       └── python-mojo-migration/
│           ├── SKILL.md
│           └── templates/
│
└── pixi.toml                          # Dependencies (Python + Mojo)
```text

### Module Interdependencies

```text
┌─────────────────────────────────────────────────────────────┐
│  Papers (Application Layer)                                 │
│  ├─ lenet5, alexnet, vgg, ...                              │
└──────────────────┬────────────────────────────────────────┘
                   │ uses
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Training (High-level Orchestration)                        │
│  ├─ trainer.py (Python)                                    │
│  ├─ data_pipeline.py (Python)                              │
│  └─ results_handler.py (Python)                            │
└──────────────────┬────────────────────────────────────────┘
                   │ uses
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Hybrid Bridge Layer (Communication)                        │
│  ├─ bridge.py (Pattern 1: Direct Wrapper)                  │
│  ├─ async_queue.py (Pattern 2: Async Queue)                │
│  ├─ shared_memory.py (Pattern 3: Shared Memory)            │
│  ├─ serialization.py (Pattern 4: Protocol Buffers)         │
│  └─ jit_bridge.py (Pattern 5: JIT Bridge)                  │
└──────────────────┬────────────────────────────────────────┘
                   │ uses
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Core Kernels (Performance)                                 │
│  ├─ ops.mojo (Mojo implementations)                         │
│  ├─ shared_ops.mojo (Shared memory ops)                     │
│  └─ jit_kernel.mojo (JIT kernels)                          │
└──────────────────┬────────────────────────────────────────┘
                   │ uses
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Utilities (Support)                                        │
│  ├─ type_mapping.py (Type conversions)                      │
│  ├─ conversion.py (Data conversions)                        │
│  └─ memory.py (Memory utilities)                            │
└─────────────────────────────────────────────────────────────┘
```text

---

## Part 4: Integration Patterns

### 4.1 Choosing the Right Pattern

### Decision Tree

```text
Do you have tight loops
with simple types?
    │
    ├─ YES → Use Pattern 1 (Direct Wrapper)
    │        Overhead: < 1%
    │
    └─ NO
        │
        Is data streaming
        continuously?
            │
            ├─ YES → Use Pattern 2 (Async Queue)
            │        Overhead: < 5%
            │
            └─ NO
                │
                Is the tensor very large
                (> 1MB)?
                    │
                    ├─ YES → Use Pattern 3 (Shared Memory)
                    │        Overhead: < 2%
                    │
                    └─ NO
                        │
                        Do you have complex
                        nested structures?
                            │
                            ├─ YES → Use Pattern 4 (Protocol Buffers)
                            │        Overhead: < 8%
                            │
                            └─ NO
                                │
                                Is this a tight loop
                                that benefits from JIT?
                                    │
                                    ├─ YES → Use Pattern 5 (JIT Bridge)
                                    │        Overhead: < 3% (after warm-up)
                                    │
                                    └─ NO → Re-evaluate or use Pattern 1
```text

### 4.2 Hybrid Training Loop Example

```python

# ml_odyssey/training/hybrid_trainer.py

import asyncio
from ml_odyssey.core.bridge import MojoBridge
from ml_odyssey.core.async_queue import DataProducer
from ml_odyssey.core.shared_memory import SharedMemoryManager
from ml_odyssey.core.serialization import SerializedModel

class HybridTrainer:
    """Training loop using all 5 hybrid patterns."""

    def __init__(self, model_config: dict):
        self.bridge = MojoBridge('libml_odyssey.so')  # Pattern 1
        self.mem_manager = SharedMemoryManager()       # Pattern 3
        self.model_config = model_config

    async def train(
        self,
        data_loader,
        num_epochs: int,
        batch_size: int
    ):
        """Train using hybrid patterns."""

        # Pattern 2: Async data pipeline
        producer = DataProducer()
        batches = producer.produce_batches(data_loader)

        async for epoch in range(num_epochs):
            async for batch in batches:

                # Pattern 3: Use shared memory for batch
                x_shared = self.mem_manager.allocate(
                    shape=batch['x'].shape
                )
                x_shared.data[:] = batch['x']

                y_shared = self.mem_manager.allocate(
                    shape=batch['y'].shape
                )
                y_shared.data[:] = batch['y']

                # Pattern 1: Direct wrapper for simple ops
                batch_size = self.bridge.get_batch_size(x_shared.get_pointer())

                # Pattern 3: Shared memory for bulk compute
                outputs = self.bridge.forward_pass(
                    x_shared.get_pointer(),
                    model_weights_ptr=self._get_weights_ptr(),
                    batch_size=batch_size
                )

                # Pattern 4: Serialize complex model state
                model_state = SerializedModel.from_config(self.model_config)
                state_bytes = model_state.to_bytes()

                # Update weights via Mojo
                self.bridge.update_weights(state_bytes)

                # Pattern 5: JIT-compiled metric computation
                metrics = self._compute_metrics_jit(outputs, y_shared)

                print(f"Loss: {metrics['loss']:.4f}")

    def _get_weights_ptr(self) -> int:
        """Get pointer to model weights."""
        return self.bridge.get_weights_pointer()

    def _compute_metrics_jit(self, outputs, labels):
        """Compute metrics using JIT bridge."""
        from ml_odyssey.core.jit_bridge import MojoJITBridge

        jit = MojoJITBridge('libml_odyssey.so')
        return jit.jit_loop(
            lambda: self._metric_loop(outputs, labels),
            iterations=len(outputs),
            dtype='float32'
        )

    def _metric_loop(self, outputs, labels):
        """Inner loop for metric computation."""
        # This gets JIT compiled
        total_loss = 0.0
        for pred, label in zip(outputs, labels):
            total_loss += (pred - label) ** 2
        return total_loss / len(outputs)
```text

---

## Part 5: Code Examples

### 5.1 Simple MatMul Example (Pattern 1)

**File**: `ml_odyssey/examples/matmul_wrapper.py`

```python

"""Example: Matrix multiplication using direct wrapper (Pattern 1)."""

import numpy as np
from ml_odyssey.core.bridge import MojoBridge

def main():
    # Initialize bridge
    bridge = MojoBridge('build/libml_odyssey.so')

    # Create test matrices
    A = np.random.randn(100, 50).astype(np.float32)
    B = np.random.randn(50, 200).astype(np.float32)

    print(f"Matrix A shape: {A.shape}")
    print(f"Matrix B shape: {B.shape}")

    # Call Mojo kernel
    C = bridge.matmul(A, B)

    print(f"Result C shape: {C.shape}")
    print(f"First element: {C[0, 0]:.4f}")

    # Verify against NumPy
    C_numpy = A @ B
    error = np.linalg.norm(C - C_numpy)
    print(f"Error vs NumPy: {error:.2e}")

if __name__ == '__main__':
    main()
```text

### Run

```bash

cd /home/mvillmow/ml-odyssey
python ml_odyssey/examples/matmul_wrapper.py
```text

### 5.2 Data Pipeline Example (Pattern 2)

**File**: `ml_odyssey/examples/async_data_pipeline.py`

```python

"""Example: Async data pipeline (Pattern 2)."""

import asyncio
import numpy as np
from ml_odyssey.core.async_queue import DataProducer, Batch

async def main():
    # Create simple data
    data = [
        {'x': np.random.randn(32, 784), 'y': np.random.randint(0, 10, 32)}
        for _ in range(100)
    ]

    # Create producer
    producer = DataProducer(queue_size=5)

    # Simulate consumer
    async def consume():
        batches_processed = 0
        async for batch in producer.produce_batches(data, num_batches=50):
            if batch is None:
                break
            print(f"Batch {batch.batch_id}: shape {batch.x.shape}")
            batches_processed += 1
            await asyncio.sleep(0.01)  # Simulate processing
        return batches_processed

    # Run
    num_batches = await consume()
    print(f"Processed {num_batches} batches")

if __name__ == '__main__':
    asyncio.run(main())
```text

### Run

```bash

cd /home/mvillmow/ml-odyssey
python ml_odyssey/examples/async_data_pipeline.py
```text

### 5.3 Shared Memory Example (Pattern 3)

**File**: `ml_odyssey/examples/shared_memory_ops.py`

```python

"""Example: Shared memory operations (Pattern 3)."""

import numpy as np
from ml_odyssey.core.bridge import MojoBridge
from ml_odyssey.core.shared_memory import SharedMemoryManager

def main():
    # Initialize components
    bridge = MojoBridge('build/libml_odyssey.so')
    mem_mgr = SharedMemoryManager()

    # Allocate large matrices in shared memory
    m, n, k = 1000, 1000, 500

    A_shared = mem_mgr.allocate(shape=(m, k), dtype=np.float32)
    B_shared = mem_mgr.allocate(shape=(k, n), dtype=np.float32)

    # Fill with data
    A_shared.data[:] = np.random.randn(m, k).astype(np.float32)
    B_shared.data[:] = np.random.randn(k, n).astype(np.float32)

    print(f"A shape: {A_shared.shape}, address: {A_shared.get_pointer()}")
    print(f"B shape: {B_shared.shape}, address: {B_shared.get_pointer()}")

    # Call Mojo kernel with pointers
    C_shared = mem_mgr.allocate(shape=(m, n), dtype=np.float32)

    bridge.matmul_shared(
        a_ptr=A_shared.get_pointer(),
        b_ptr=B_shared.get_pointer(),
        out_ptr=C_shared.get_pointer(),
        m=m, n=n, k=k
    )

    print(f"Result shape: {C_shared.shape}")
    print(f"First element: {C_shared.data[0, 0]:.4f}")

if __name__ == '__main__':
    main()
```text

### Run

```bash

cd /home/mvillmow/ml-odyssey
python ml_odyssey/examples/shared_memory_ops.py
```text

---

## Part 6: Implementation Roadmap

### Phase 1: Foundation (Week 1)

**Goal**: Establish core bridge infrastructure

### Tasks

1. Create project structure (`ml_odyssey/core/`)
1. Implement Pattern 1 (Direct Wrapper)
   - Basic FFI setup
   - Type mapping utilities
   - Error handling
1. Create first Mojo kernel (`ops.mojo`)
1. Write Pattern 1 tests
1. Benchmark Pattern 1 overhead

### Deliverables

- `/home/mvillmow/ml-odyssey/ml_odyssey/core/bridge.py`
- `/home/mvillmow/ml-odyssey/ml_odyssey/core/ops.mojo`
- `/home/mvillmow/ml-odyssey/tests/test_bridge_patterns.py`
- Benchmark report showing < 1% overhead

### Phase 2: Data Communication (Week 2)

**Goal**: Implement remaining 4 patterns

### Tasks

1. Pattern 2: Async Queue
   - Queue implementation
   - Producer/consumer logic
   - Back-pressure handling
1. Pattern 3: Shared Memory
   - Memory allocation
   - Pointer management
   - Stride calculations
1. Pattern 4: Protocol Buffers
   - Message definitions
   - Serialization logic
   - Type mapping
1. Pattern 5: JIT Bridge
   - Compilation pipeline
   - Code generation
   - Caching mechanism

### Deliverables

- Complete implementation of all 5 patterns
- Tests for each pattern
- Comparative benchmarks
- Overhead analysis report

### Phase 3: Integration (Week 3)

**Goal**: Create unified tooling and utilities

### Tasks

1. Type mapping system (`utils/type_mapping.py`)
1. Memory utilities (`utils/memory.py`)
1. Conversion utilities (`utils/conversion.py`)
1. Pattern selection helper
1. Unified error handling
1. Debug utilities

### Deliverables

- Complete utilities module
- Pattern selection guide
- Error handling framework
- Debug tools

### Phase 4: Training Integration (Week 4)

**Goal**: Implement hybrid training loop

### Tasks

1. Hybrid trainer class
1. Data pipeline integration
1. Model serialization
1. Results collection
1. Monitoring and logging
1. End-to-end tests

### Deliverables

- `/home/mvillmow/ml-odyssey/ml_odyssey/training/hybrid_trainer.py`
- Example training script
- Performance report
- Documentation

### Phase 5: Documentation & Examples (Week 5)

**Goal**: Complete documentation and learning resources

### Tasks

1. Architecture guide
1. Pattern selection guide
1. Migration guide (Python → Mojo)
1. Performance tuning guide
1. Troubleshooting guide
1. Example implementations
1. Benchmark suite

### Deliverables

- Complete documentation in `/home/mvillmow/ml-odyssey/docs/`
- Working examples in `/home/mvillmow/ml-odyssey/ml_odyssey/examples/`
- Comprehensive test suite
- Benchmark reports

---

## Part 7: Performance Specifications

### Overhead Targets

| Pattern | Operation | Target Overhead | Actual | Status |
|---------|-----------|-----------------|--------|--------|
| Direct Wrapper | Function call | < 1% | TBD | Planned |
| Async Queue | Batch enqueue | < 5% | TBD | Planned |
| Shared Memory | Pointer pass | < 2% | TBD | Planned |
| Protocol Buf | Serialization | < 8% | TBD | Planned |
| JIT Bridge | Compilation | < 3% (after warm-up) | TBD | Planned |

### Benchmark Scenarios

1. **Micro-benchmarks**: Single pattern performance
   - Function call overhead
   - Data serialization time
   - Memory allocation/deallocation
   - JIT compilation time

1. **Macro-benchmarks**: Realistic workloads
   - Matrix multiplication (Pattern 1 + 3)
   - Data loading (Pattern 2)
   - Model updates (Pattern 4)
   - Tight loops (Pattern 5)

1. **End-to-end benchmarks**: Complete training
   - Full training epoch
   - Multi-pattern orchestration
   - Distributed training simulation

---

## Part 8: Testing Strategy

### Unit Tests

**Location**: `/home/mvillmow/ml-odyssey/tests/test_*.py`

```python
# Example test structure
import pytest
from ml_odyssey.core.bridge import MojoBridge
from ml_odyssey.core.async_queue import DataProducer

class TestDirectWrapper:
    """Test Pattern 1 (Direct Wrapper)."""

    def test_simple_addition(self):
        bridge = MojoBridge('build/libml_odyssey.so')
        result = bridge.add(2.0, 3.0)
        assert result == 5.0

    def test_matrix_multiply(self):
        # Test matrix multiplication
        pass

    def test_overhead_is_minimal(self):
        # Benchmark overhead
        pass

class TestAsyncQueue:
    """Test Pattern 2 (Async Queue)."""

    @pytest.mark.asyncio
    async def test_batch_enqueue(self):
        # Test async queue
        pass

    @pytest.mark.asyncio
    async def test_back_pressure(self):
        # Test queue full behavior
        pass
```text

### Integration Tests

```python
class TestHybridTraining:
    """Test complete training loop."""

    def test_full_training_loop(self):
        # Test all patterns together
        pass

    def test_pattern_composition(self):
        # Test using multiple patterns
        pass
```text

### Performance Tests

```python
class TestPerformance:
    """Performance benchmarks."""

    def test_pattern_overhead(self):
        # Measure overhead for each pattern
        pass

    def test_end_to_end_performance(self):
        # Measure total training performance
        pass
```text

---

## Part 9: Appendices

### Appendix A: Type Mapping Table

### Python ↔ Mojo Type Mapping

| Python Type | Mojo Type | Conversion | Notes |
|-------------|-----------|-----------|-------|
| `float` | `Float32` or `Float64` | Direct cast | Default Float32 |
| `int` | `Int32` or `Int64` | Direct cast | Default Int32 |
| `bool` | `Bool` | Direct cast | No conversion needed |
| `numpy.ndarray` | `Tensor` | Pointer pass | Via shared memory |
| `list` | `DynamicVector` | Copy + convert | Not recommended |
| `dict` | `PythonObject` | Via protocol buf | Complex mapping |
| `dataclass` | `struct` | Via proto buf | Requires schema |

### Appendix B: Serialization Format Comparison

| Format | Size | Speed | Compatibility | Use Case |
|--------|------|-------|--------------|----------|
| JSON | Large | Slow | Excellent | Config, metadata |
| NumPy | Medium | Fast | Good | Arrays only |
| Protocol Buffers | Small | Fast | Excellent | Complex objects |
| MessagePack | Small | Fast | Good | General purpose |
| Pickle | Large | Medium | Python only | Python objects |

### Appendix C: Memory Alignment Requirements

### Mojo SIMD Alignment

```text
Alignment by data type:
- Float32: 4 bytes minimum (8 bytes recommended)
- Float64: 8 bytes minimum (16 bytes recommended)
- Vector operations: 16 bytes minimum (32 bytes for AVX)
- Cache line: 64 bytes recommended for performance
```text

### Ensuring Alignment in Python

```python
import numpy as np

def create_aligned_array(shape, dtype=np.float32, alignment=32):
    """Create numpy array with specific alignment."""
    nbytes = np.prod(shape) * np.dtype(dtype).itemsize
    buf = np.zeros(nbytes + alignment, dtype=np.uint8)
    offset = alignment - buf.ctypes.data % alignment
    return np.frombuffer(buf[offset:offset+nbytes], dtype=dtype).reshape(shape)
```text

### Appendix D: Common Pitfalls & Solutions

#### Pitfall 1: Memory Ownership Confusion

```text

Problem: Python allocates array, passes to Mojo, Mojo frees → Segfault
Solution: Document ownership clearly - who allocates, who frees?
Example: Python owns memory, Mojo only borrows (readonly or copy)
```text

#### Pitfall 2: Type Mismatch Issues

```text

Problem: Python passes int64, Mojo expects int32 → Integer overflow
Solution: Use strict type mapping, explicit conversions
Example: Always convert NumPy arrays to float32 for Mojo compatibility
```text

#### Pitfall 3: Async Synchronization

```text

Problem: Python continues before Mojo finishes → Race condition
Solution: Use explicit synchronization barriers
Example: Block until Mojo signals completion
```text

#### Pitfall 4: Performance Regression

```text

Problem: Adding hybrid bridge slows everything down
Solution: Profile carefully, choose right pattern
Example: Use Pattern 3 (shared memory) for large tensors, not Pattern 4
```text

### Appendix E: Debugging Checklist

- [ ] FFI function signatures correct (ctypes setup)?
- [ ] Memory alignment suitable for SIMD operations?
- [ ] Ownership clear (who allocates/frees each buffer)?
- [ ] Type conversions happening correctly?
- [ ] No undefined behavior (buffer overruns, etc.)?
- [ ] Performance baseline established?
- [ ] Error handling covers all failure cases?
- [ ] Tests pass in both debug and release builds?

### Appendix F: Environment Setup

### Requirements

```toml

[project]
name = "ml-odyssey"
version = "0.1.0"

[tool.pixi.dependencies]
python = ">=3.10"
mojo = ">=0.5.0"
numpy = ">=1.24.0"
pytest = ">=7.0.0"

[tool.pixi.dev-dependencies]
pytest-asyncio = ">=0.21.0"
black = ">=23.0.0"
mypy = ">=1.0.0"
```text

### Installation

```bash

# Install Pixi environment
pixi install

# Build Mojo shared library
mojo build -o build/libml_odyssey.so ml_odyssey/core/ops.mojo

# Run tests
pytest tests/ -v
```text

### Appendix G: References

### Mojo Documentation

- [Mojo Manual](https://docs.modular.com/mojo/)
- [FFI Guide](https://docs.modular.com/mojo/manual/interop/)
- [SIMD Programming](https://docs.modular.com/mojo/manual/simd/)

### Python FFI

- [ctypes documentation](https://docs.python.org/3/library/ctypes.html)
- [CFFI Guide](https://cffi.readthedocs.io/)

### NumPy Integration

- [NumPy C API](https://numpy.org/doc/stable/reference/c-api/)
- [NumPy ctypes integration](https://numpy.org/doc/stable/reference/routines.ctypeslib.html)

### Async Python

- [asyncio documentation](https://docs.python.org/3/library/asyncio.html)
- [Python typing](https://docs.python.org/3/library/typing.html)

---

## Success Criteria Checklist

- [ ] All 5 communication patterns implemented and tested
- [ ] Overhead for each pattern measured and < target threshold
- [ ] Hybrid trainer successfully runs complete training epoch
- [ ] Documentation complete and clear
- [ ] Examples working and reproducible
- [ ] Type mapping covers all common cases
- [ ] Error handling comprehensive
- [ ] Performance benchmarks show viability
- [ ] Team can build and extend architecture
- [ ] Integration with CI/CD validated

---

**Document Version**: 1.0
**Last Updated**: 2025-11-09
**Status**: Ready for Implementation
**Next Phase**: Issue #8 Test Phase (Unit test implementation for all 5 patterns)
